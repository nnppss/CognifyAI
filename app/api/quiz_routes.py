from flask import Blueprint, flash, redirect, render_template, request, url_for

from app.api.helpers import build_qa_redirect, flash_generation_failure, load_chunks, load_lecture_metadata, mark_lecture_opened
from app.api.quiz_utils import evaluate_multi_select, format_answer_list, normalize_text, prepare_quiz_questions

quiz_bp = Blueprint("quiz", __name__)


@quiz_bp.route("/quiz", methods=["GET", "POST"])
def quiz_view():
    lecture_id = (request.values.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("lectures.index"))

    questions = []
    results = []
    total_score = None
    coaching = {}

    try:
        from app.core.quiz_engine import evaluate_short_answer, generate_quiz

        chunks = load_chunks(lecture_id)
        force = request.method == "POST" and request.form.get("action") == "regenerate"
        quiz = generate_quiz(lecture_id, chunks, force=force)
        questions = prepare_quiz_questions(quiz.get("questions", []))
    except Exception as exc:
        flash_generation_failure("Quiz generation", exc)
        return build_qa_redirect(lecture_id)

    if request.method == "POST" and request.form.get("action") == "submit":
        total = 0.0
        possible = 0.0
        for idx, question in enumerate(questions, start=1):
            qtype = question.get("type", "")
            answer_key = f"q_{idx}"

            selected_answers = []
            if qtype == "multi_select":
                selected_answers = [
                    answer.strip() for answer in request.form.getlist(answer_key) if answer and answer.strip()
                ]
            else:
                single_answer = (request.form.get(answer_key) or "").strip()
                if single_answer:
                    selected_answers = [single_answer]

            correct = str(question.get("correct", "")).strip()
            correct_options = question.get("resolved_correct_options", [])
            if qtype in {"mcq", "multi_select"}:
                correct_display = format_answer_list(correct_options or [correct])
            else:
                correct_display = correct or "No answer"

            entry = {
                "question": question.get("question", ""),
                "type": qtype,
                "user_answer": format_answer_list(selected_answers),
                "correct": correct_display,
                "explanation": question.get("explanation", ""),
                "score": 0.0,
                "feedback": "",
                "assessment": "No answer",
            }

            if qtype == "mcq":
                expected = correct_options[0] if correct_options else correct
                if selected_answers and normalize_text(selected_answers[0]) == normalize_text(expected):
                    entry["score"] = 1.0
                elif selected_answers and expected:
                    entry["feedback"] = "Selected option is incorrect."
                elif not expected:
                    entry["feedback"] = "No expected answer was configured for this question."

                possible += 1.0
                total += entry["score"]
            elif qtype == "multi_select":
                entry["score"], entry["feedback"] = evaluate_multi_select(selected_answers, correct_options)
                possible += 1.0
                total += entry["score"]
            elif qtype == "true_false":
                expected = correct
                if selected_answers and normalize_text(selected_answers[0]) == normalize_text(expected):
                    entry["score"] = 1.0
                elif selected_answers:
                    entry["feedback"] = "Selected option is incorrect."
                possible += 1.0
                total += entry["score"]
            elif qtype == "short_answer":
                user_answer = selected_answers[0] if selected_answers else ""
                entry["user_answer"] = user_answer or "No answer"
                possible += 1.0
                if not user_answer:
                    entry["score"] = 0.0
                    entry["feedback"] = "No response submitted."
                else:
                    try:
                        eval_result = evaluate_short_answer(entry["question"], correct, user_answer)
                        entry["score"] = float(eval_result.get("score", 0.0))
                        entry["feedback"] = str(eval_result.get("feedback", ""))
                    except Exception:
                        entry["score"] = 0.0
                        entry["feedback"] = "Could not auto-evaluate this answer."
                total += entry["score"]

            if entry["user_answer"] == "No answer":
                entry["assessment"] = "No answer"
            elif entry["score"] >= 0.99:
                entry["assessment"] = "Correct"
            elif entry["score"] <= 0.01:
                entry["assessment"] = "Incorrect"
            else:
                entry["assessment"] = "Partially correct"

            results.append(entry)

        total_score = 0.0 if possible == 0 else round((total / possible) * 100.0, 2)
        try:
            from app.core.coaching_engine import record_quiz_attempt

            attempt = record_quiz_attempt(lecture_id, results, total_score)
            coaching = attempt.get("coaching", {})
        except Exception as exc:
            flash(f"Coaching insights could not be updated: {exc}", "error")
    else:
        try:
            from app.core.coaching_engine import build_coaching_payload

            coaching = build_coaching_payload(lecture_id)
        except Exception:
            coaching = {}

    return render_template(
        "quiz.html",
        lecture_id=lecture_id,
        questions=questions,
        results=results,
        total_score=total_score,
        lecture_meta=mark_lecture_opened(lecture_id) or load_lecture_metadata(lecture_id),
        coaching=coaching,
    )
