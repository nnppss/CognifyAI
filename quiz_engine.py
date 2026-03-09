import json
import os
import re
from typing import Dict, List

from config import QUIZ_DIR, QUIZ_NUM_QUESTIONS
from llm_client import call_llm


def _chunks_to_text(chunks: List[Dict], limit: int = 70) -> str:
    clipped = chunks[:limit]
    lines = []
    for chunk in clipped:
        text = str(chunk.get("text", "")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_json_block(text: str) -> Dict:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No valid JSON object found in model response.")


def _has_multi_select_questions(payload: Dict) -> bool:
    questions = payload.get("questions", []) if isinstance(payload, dict) else []
    for question in questions:
        if not isinstance(question, dict):
            continue

        qtype = str(question.get("type", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if qtype in {"multi_select", "multiple_correct", "multiple_answers", "multi_choice", "msq"}:
            return True

        correct = question.get("correct")
        if isinstance(correct, list) and len(correct) > 1:
            return True
    return False


def generate_quiz(lecture_id: str, chunks: List[Dict], num_questions: int = QUIZ_NUM_QUESTIONS, force: bool = False) -> Dict:
    os.makedirs(QUIZ_DIR, exist_ok=True)
    quiz_path = os.path.join(QUIZ_DIR, f"{lecture_id}_quiz.json")
    if os.path.exists(quiz_path) and not force:
        with open(quiz_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if _has_multi_select_questions(cached):
            return cached

    system_prompt = "You are an exam question setter for engineering students."
    user_prompt = (
        f"Generate exactly {num_questions} questions from the lecture transcript. "
        "Mix types: 40% mcq (single correct), 20% multi_select (multiple correct), 20% true_false, 20% short_answer. "
        "Include at least 2 multi_select questions. "
        "Return only strict JSON with schema:\n"
        "{\"questions\":[{\"id\":1,\"type\":\"mcq|multi_select|true_false|short_answer\",\"question\":\"...\","
        "\"options\":[\"...\"],\"correct\":\"...\" or [\"...\"],\"explanation\":\"...\"}]}\n"
        "For mcq, correct must be exactly one answer (label like A/B/C/D or exact option text).\n"
        "For multi_select, correct must be an array with at least 2 correct answers (labels or exact option text).\n"
        "For true_false, correct must be 'True' or 'False'.\n"
        "For short_answer, keep correct concise (1-3 lines).\n\n"
        f"Transcript:\n{_chunks_to_text(chunks)}"
    )
    raw = call_llm(
        system_prompt,
        user_prompt,
        max_output_tokens=2048,
        temperature=0.2,
        task_type="quiz",
    )
    parsed = _extract_json_block(raw)

    if not _has_multi_select_questions(parsed):
        retry_prompt = (
            "The following quiz JSON does not satisfy constraints.\n"
            "Rewrite it so it includes at least 2 multi_select questions with multiple correct answers.\n"
            f"Keep exactly {num_questions} questions and return strict JSON only.\n\n"
            f"Current JSON:\n{raw}"
        )
        retry_raw = call_llm(
            system_prompt,
            retry_prompt,
            max_output_tokens=2048,
            temperature=0.2,
            task_type="quiz",
        )
        parsed = _extract_json_block(retry_raw)

    parsed["lecture_id"] = lecture_id

    with open(quiz_path, "w", encoding="utf-8") as handle:
        json.dump(parsed, handle, ensure_ascii=False, indent=2)
    return parsed


def evaluate_short_answer(question: str, expected_answer: str, user_answer: str) -> Dict:
    system_prompt = "You are a strict but fair exam evaluator."
    user_prompt = (
        "Evaluate the student's short answer and return JSON only with keys: score, feedback.\n"
        "score should be from 0 to 1.\n\n"
        f"Question: {question}\n"
        f"Expected: {expected_answer}\n"
        f"Student: {user_answer}"
    )
    raw = call_llm(
        system_prompt,
        user_prompt,
        max_output_tokens=256,
        temperature=0.0,
        task_type="quiz_eval",
    )
    parsed = _extract_json_block(raw)
    score = float(parsed.get("score", 0.0))
    parsed["score"] = max(0.0, min(1.0, score))
    parsed["feedback"] = str(parsed.get("feedback", ""))
    return parsed
