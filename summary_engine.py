import json
import os
from typing import Dict, List

from config import SUMMARY_DIR, SUMMARY_MAX_WORDS
from llm_client import call_llm


def _chunks_to_text(chunks: List[Dict], include_timestamps: bool = False) -> str:
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        if include_timestamps:
            lines.append(
                f"[{idx}] {float(chunk.get('start', 0.0)):.1f}-{float(chunk.get('end', 0.0)):.1f}s: {text}"
            )
        else:
            lines.append(text)
    return "\n".join(lines)


def _slice_chunks(chunks: List[Dict], batch_size: int = 30) -> List[List[Dict]]:
    return [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]


def _save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_summary(lecture_id: str, chunks: List[Dict], force: bool = False) -> Dict:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary_path = os.path.join(SUMMARY_DIR, f"{lecture_id}_summary.json")
    if os.path.exists(summary_path) and not force:
        return _load_json(summary_path)

    batches = _slice_chunks(chunks)
    partials = []

    system_prompt = "You are an expert academic summarizer."
    for batch in batches:
        user_prompt = (
            "Create concise markdown summary bullets for these lecture segments. "
            "Capture the lecture flow, key ideas, and exam-relevant points.\n\n"
            f"Segments:\n{_chunks_to_text(batch)}"
        )
        partials.append(
            call_llm(
                system_prompt,
                user_prompt,
                max_output_tokens=1024,
                temperature=0.2,
                task_type="summary",
            )
        )

    user_prompt_final = (
        f"Merge these partial summaries into one polished final summary under {SUMMARY_MAX_WORDS} words. "
        "Return only markdown with this exact structure:\n"
        "# Lecture Summary\n"
        "## Big Picture\n"
        "## Key Concepts\n"
        "## High-Yield Takeaways\n"
        "## Quick Revision Checklist\n"
        "Use concise bullets and bold the most important terms.\n\n"
        "Partial summaries:\n"
        + "\n\n".join(partials)
    )
    final_summary = call_llm(
        system_prompt,
        user_prompt_final,
        max_output_tokens=1024,
        temperature=0.2,
        task_type="summary",
    )

    payload = {
        "lecture_id": lecture_id,
        "summary": final_summary.strip(),
        "partials": partials,
    }
    _save_json(summary_path, payload)
    return payload


def generate_detailed_notes(lecture_id: str, chunks: List[Dict], force: bool = False) -> Dict:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    notes_path = os.path.join(SUMMARY_DIR, f"{lecture_id}_notes.json")
    if os.path.exists(notes_path) and not force:
        return _load_json(notes_path)

    batches = _slice_chunks(chunks)
    partials = []

    system_prompt = "You are an expert lecture note writer for exam preparation."
    for batch in batches:
        user_prompt = (
            "Create detailed, exam-ready markdown notes from these lecture segments. "
            "Focus on concepts, definitions, explanations, and examples.\n\n"
            f"Segments:\n{_chunks_to_text(batch)}"
        )
        partials.append(
            call_llm(
                system_prompt,
                user_prompt,
                max_output_tokens=1536,
                temperature=0.2,
                task_type="notes",
            )
        )

    user_prompt_final = (
        "Merge these partial notes into one cohesive final set of lecture notes.\n"
        "Structure exactly with markdown headings:\n"
        "# Lecture Notes\n"
        "## Concept Map\n"
        "## Important Definitions\n"
        "## Step-by-Step Explanations\n"
        "## Worked Insights and Examples\n"
        "## Common Pitfalls\n"
        "## Exam-Focused Revision Checklist\n"
        "Remove repetition, keep the content factual and revision-friendly, and bold key terms/formulas.\n\n"
        "Partial notes:\n"
        + "\n\n".join(partials)
    )
    notes = call_llm(
        system_prompt,
        user_prompt_final,
        max_output_tokens=4096,
        temperature=0.2,
        task_type="notes",
    )

    payload = {
        "lecture_id": lecture_id,
        "notes": notes.strip(),
    }
    _save_json(notes_path, payload)
    return payload
