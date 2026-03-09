import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from config import MAX_CONTENT_LENGTH, TRANSCRIPT_DIR
from indexing import build_index
from media_utils import (
    allowed_file,
    create_upload_lecture_id,
    download_youtube_audio,
    download_youtube_video_for_frames,
    extract_audio_ffmpeg,
    get_youtube_video_language,
    load_lecture_metadata,
    save_lecture_metadata,
    save_uploaded_video,
)
from frame_ocr_utils import extract_frame_ocr_segments
from pdf_generator import generate_pdf
from qa_engine import LectureQA
from quiz_engine import evaluate_short_answer, generate_quiz
from speechmatics_transcribe import transcribe_audio_speechmatics
from summary_engine import generate_detailed_notes, generate_summary
from transcript_utils import get_video_id, merge_segments, save_chunks
from youtube_transcript_utils import fetch_youtube_transcript

app = Flask(__name__)
app.secret_key = "cognifyai_sem7_secret"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _load_chunks(lecture_id: str):
    path = os.path.join(TRANSCRIPT_DIR, f"{lecture_id}_chunks.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lecture '{lecture_id}' has not been processed yet.")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_qa_redirect(lecture_id: str):
    meta = load_lecture_metadata(lecture_id)
    src_url = meta.get("source_url", "")
    return redirect(url_for("qa", lecture_id=lecture_id, src_url=src_url))


_MULTI_SELECT_TYPES = {
    "multi_select",
    "multiple_correct",
    "multiple_answers",
    "multi_choice",
    "msq",
}
_CHOICE_SPLIT_PATTERN = re.compile(r"\s*(?:,|/|;|\||\band\b|&)\s*", flags=re.IGNORECASE)


def _normalize_quiz_type(raw_type: str) -> str:
    qtype = str(raw_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if qtype in _MULTI_SELECT_TYPES:
        return "multi_select"
    if qtype in {"mcq", "multiple_choice", "single_choice"}:
        return "mcq"
    if qtype in {"true_false", "true/false", "boolean"}:
        return "true_false"
    if qtype in {"short_answer", "short", "open_ended", "openended"}:
        return "short_answer"
    return qtype


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def _clean_choice_options(raw_options) -> list:
    if not isinstance(raw_options, list):
        return []
    options = []
    for option in raw_options:
        text = str(option or "").strip()
        if text:
            options.append(text)
    return options


def _token_to_option(token: str, options: list) -> str:
    probe = str(token or "").strip()
    if not probe:
        return ""

    probe_key = _normalize_text(probe)
    for option in options:
        if _normalize_text(option) == probe_key:
            return option

    letter_match = re.fullmatch(r"\(?\s*([A-Za-z])\s*[\).]?\s*", probe)
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]

    number_match = re.fullmatch(r"\(?\s*(\d+)\s*[\).]?\s*", probe)
    if number_match:
        idx = int(number_match.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]

    prefixed_match = re.match(r"^\s*([A-Za-z]|\d+)\s*[\).:\-]\s*(.+?)\s*$", probe)
    if prefixed_match:
        by_label = _token_to_option(prefixed_match.group(1), options)
        if by_label:
            return by_label
        trailing = prefixed_match.group(2)
        trailing_key = _normalize_text(trailing)
        for option in options:
            if _normalize_text(option) == trailing_key:
                return option

    return ""


def _extract_answer_tokens(raw_answer, options: list) -> list:
    if isinstance(raw_answer, list):
        return [str(item).strip() for item in raw_answer if str(item).strip()]

    text = str(raw_answer or "").strip()
    if not text:
        return []

    if options and _token_to_option(text, options):
        return [text]

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    tokens = [part.strip() for part in _CHOICE_SPLIT_PATTERN.split(text) if part.strip()]
    if len(tokens) > 1:
        return tokens
    return [text]


def _resolve_choice_answers(raw_answer, options: list) -> list:
    tokens = _extract_answer_tokens(raw_answer, options)
    resolved = []
    seen = set()

    for token in tokens:
        option = _token_to_option(token, options)
        candidate = option or str(token).strip()
        key = _normalize_text(candidate)
        if key and key not in seen:
            seen.add(key)
            resolved.append(candidate)

    return resolved


def _format_answer_list(answers: list) -> str:
    cleaned = [str(answer).strip() for answer in answers if str(answer).strip()]
    return ", ".join(cleaned) if cleaned else "No answer"


def _prepare_quiz_questions(raw_questions: list) -> list:
    prepared = []
    for raw_question in raw_questions:
        question = dict(raw_question or {})
        qtype = _normalize_quiz_type(question.get("type", ""))
        options = _clean_choice_options(question.get("options", []))
        question["options"] = options

        resolved_correct = []
        if qtype in {"mcq", "multi_select"}:
            resolved_correct = _resolve_choice_answers(question.get("correct", ""), options)
            if qtype == "mcq" and len(resolved_correct) > 1:
                qtype = "multi_select"

        question["type"] = qtype
        question["resolved_correct_options"] = resolved_correct
        prepared.append(question)
    return prepared


def _evaluate_multi_select(selected_answers: list, correct_answers: list) -> tuple:
    selected_norm = {_normalize_text(answer) for answer in selected_answers if _normalize_text(answer)}
    correct_norm = {_normalize_text(answer) for answer in correct_answers if _normalize_text(answer)}

    if not correct_norm:
        return 0.0, "No correct answers were configured for this question."
    if not selected_norm:
        return 0.0, ""
    if selected_norm == correct_norm:
        return 1.0, ""

    if selected_norm.issubset(correct_norm):
        missing = [answer for answer in correct_answers if _normalize_text(answer) not in selected_norm]
        feedback = f"Missing: {', '.join(missing)}." if missing else ""
        score = len(selected_norm) / len(correct_norm)
        return round(score, 2), feedback

    wrong = [answer for answer in selected_answers if _normalize_text(answer) not in correct_norm]
    missing = [answer for answer in correct_answers if _normalize_text(answer) not in selected_norm]
    parts = []
    if wrong:
        parts.append(f"Not correct: {', '.join(wrong)}")
    if missing:
        parts.append(f"Missing: {', '.join(missing)}")
    feedback = ". ".join(parts)
    if feedback:
        feedback += "."
    return 0.0, feedback


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        source_type = (request.form.get("source_type") or "youtube").strip()

        try:
            transcript_source = ""
            detected_language = ""
            ocr_error = ""
            ocr_segments = []
            video_path = ""

            if source_type == "upload":
                upload = request.files.get("video_file")
                if not upload or not upload.filename:
                    raise ValueError("Please choose a local video file.")
                if not allowed_file(upload.filename):
                    raise ValueError("Unsupported file format. Allowed: MP4, MKV, AVI, MOV.")

                lecture_id = create_upload_lecture_id()
                video_path = save_uploaded_video(upload, lecture_id)
                audio_path = extract_audio_ffmpeg(video_path, lecture_id)
                source_url = ""
                source_label = upload.filename
                raw_segments = transcribe_audio_speechmatics(audio_path)
                transcript_source = "speechmatics-local"
            else:
                youtube_url = (request.form.get("youtube_url") or "").strip()
                if not youtube_url:
                    raise ValueError("Please paste a YouTube lecture URL.")
                video_id = get_video_id(youtube_url)
                lecture_id = f"yt_{video_id}"
                source_url = youtube_url
                source_label = youtube_url

                try:
                    video_path = download_youtube_video_for_frames(youtube_url, lecture_id)
                except Exception as exc:
                    video_path = ""
                    ocr_error = f"Frame analysis skipped: {exc}"

                try:
                    detected_language = get_youtube_video_language(youtube_url)
                except Exception:
                    detected_language = ""

                try:
                    raw_segments, transcript_language = fetch_youtube_transcript(
                        video_id,
                        preferred_language=detected_language,
                    )
                    if transcript_language:
                        detected_language = transcript_language
                    transcript_source = "youtube-transcript"
                except Exception:
                    fallback_language = detected_language or "auto"
                    audio_path = download_youtube_audio(youtube_url, lecture_id)
                    raw_segments = transcribe_audio_speechmatics(
                        audio_path,
                        language=fallback_language,
                    )
                    transcript_source = "speechmatics-youtube-fallback"
                    if not detected_language:
                        detected_language = fallback_language

            if video_path:
                try:
                    ocr_segments = extract_frame_ocr_segments(video_path, lecture_id)
                except Exception as exc:
                    ocr_segments = []
                    ocr_error = f"Frame analysis skipped: {exc}"

            if ocr_segments:
                raw_segments.extend(ocr_segments)
                raw_segments.sort(
                    key=lambda seg: (
                        float(seg.get("start", 0.0)),
                        float(seg.get("end", seg.get("start", 0.0))),
                    )
                )
                transcript_source = f"{transcript_source}+ocr"

            chunks = merge_segments(raw_segments)
            save_chunks(lecture_id, chunks)
            build_index(lecture_id)

            save_lecture_metadata(
                lecture_id,
                {
                    "lecture_id": lecture_id,
                    "source_type": source_type,
                    "source_label": source_label,
                    "source_url": source_url,
                    "transcript_source": transcript_source,
                    "detected_language": detected_language,
                    "ocr_segment_count": len(ocr_segments),
                    "ocr_error": ocr_error,
                    "chunk_count": len(chunks),
                },
            )
        except Exception as exc:
            flash(f"Processing failed: {exc}", "error")
            return redirect(url_for("index"))

        flash(f"Lecture processed successfully: {lecture_id}, chunks={len(chunks)}", "success")
        return _build_qa_redirect(lecture_id)

    return render_template("index.html")


@app.route("/qa", methods=["GET", "POST"])
def qa():
    lecture_id = (request.args.get("lecture_id") or "").strip()
    src_url = (request.args.get("src_url") or "").strip()

    answer = None
    segments = []
    timestamp = None
    mode = None
    question = ""
    top_k = 3

    if request.method == "POST":
        lecture_id = (request.form.get("lecture_id") or "").strip()
        src_url = (request.form.get("src_url") or "").strip()
        question = (request.form.get("question") or "").strip()
        try:
            top_k = int(request.form.get("top_k") or "3")
        except ValueError:
            top_k = 3

        if not lecture_id:
            flash("Lecture ID is required.", "error")
            return redirect(url_for("qa"))
        if not question:
            flash("Please enter a question.", "error")
            return _build_qa_redirect(lecture_id)

        try:
            engine = LectureQA(lecture_id)
            result = engine.answer_question(question, top_k=top_k)
            answer = result["answer"]
            segments = result["segments"]
            timestamp = int(result["timestamp"])
            mode = result.get("mode", "rag")
        except Exception as exc:
            flash(f"Error during question answering: {exc}", "error")
            return _build_qa_redirect(lecture_id)

    has_chunks = False
    if lecture_id:
        chunks_file = os.path.join(TRANSCRIPT_DIR, f"{lecture_id}_chunks.json")
        has_chunks = os.path.exists(chunks_file)

    lecture_meta = load_lecture_metadata(lecture_id) if lecture_id else {}

    return render_template(
        "qa.html",
        lecture_id=lecture_id,
        src_url=src_url,
        question=question,
        answer=answer,
        segments=segments,
        timestamp=timestamp,
        mode=mode,
        has_chunks=has_chunks,
        top_k=top_k,
        lecture_meta=lecture_meta,
    )


@app.route("/summary")
def summary_view():
    lecture_id = (request.args.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("index"))
    try:
        chunks = _load_chunks(lecture_id)
        payload = generate_summary(lecture_id, chunks)
    except Exception as exc:
        flash(f"Summary generation failed: {exc}", "error")
        return _build_qa_redirect(lecture_id)

    return render_template(
        "summary.html",
        lecture_id=lecture_id,
        title="Lecture Summary",
        content=payload.get("summary", ""),
        mode="summary",
        lecture_meta=load_lecture_metadata(lecture_id),
    )


@app.route("/notes")
def notes_view():
    lecture_id = (request.args.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("index"))
    try:
        chunks = _load_chunks(lecture_id)
        payload = generate_detailed_notes(lecture_id, chunks)
    except Exception as exc:
        flash(f"Notes generation failed: {exc}", "error")
        return _build_qa_redirect(lecture_id)

    return render_template(
        "summary.html",
        lecture_id=lecture_id,
        title="Detailed Notes",
        content=payload.get("notes", ""),
        mode="notes",
        lecture_meta=load_lecture_metadata(lecture_id),
    )


@app.route("/download/<doc_type>")
def download_doc(doc_type: str):
    lecture_id = (request.args.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("index"))
    if doc_type not in {"summary", "notes"}:
        flash("Invalid document type.", "error")
        return _build_qa_redirect(lecture_id)

    try:
        chunks = _load_chunks(lecture_id)
        if doc_type == "summary":
            payload = generate_summary(lecture_id, chunks)
            text = payload.get("summary", "")
            title = "Lecture Summary"
        else:
            payload = generate_detailed_notes(lecture_id, chunks)
            text = payload.get("notes", "")
            title = "Detailed Notes"

        pdf_path = generate_pdf(lecture_id, title, text, doc_type)
        return send_file(pdf_path, as_attachment=True)
    except Exception as exc:
        flash(f"PDF generation failed: {exc}", "error")
        return _build_qa_redirect(lecture_id)


@app.route("/quiz", methods=["GET", "POST"])
def quiz_view():
    lecture_id = (request.values.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("index"))

    questions = []
    results = []
    total_score = None

    try:
        chunks = _load_chunks(lecture_id)
        force = request.method == "POST" and request.form.get("action") == "regenerate"
        quiz = generate_quiz(lecture_id, chunks, force=force)
        questions = _prepare_quiz_questions(quiz.get("questions", []))
    except Exception as exc:
        flash(f"Quiz generation failed: {exc}", "error")
        return _build_qa_redirect(lecture_id)

    if request.method == "POST" and request.form.get("action") == "submit":
        total = 0.0
        possible = 0.0
        for idx, q in enumerate(questions, start=1):
            qtype = q.get("type", "")
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

            correct = str(q.get("correct", "")).strip()
            correct_options = q.get("resolved_correct_options", [])
            if qtype in {"mcq", "multi_select"}:
                correct_display = _format_answer_list(correct_options or [correct])
            else:
                correct_display = correct or "No answer"

            entry = {
                "question": q.get("question", ""),
                "type": qtype,
                "user_answer": _format_answer_list(selected_answers),
                "correct": correct_display,
                "explanation": q.get("explanation", ""),
                "score": 0.0,
                "feedback": "",
                "assessment": "No answer",
            }

            if qtype == "mcq":
                expected = correct_options[0] if correct_options else correct
                if selected_answers and _normalize_text(selected_answers[0]) == _normalize_text(expected):
                    entry["score"] = 1.0
                elif selected_answers and expected:
                    entry["feedback"] = "Selected option is incorrect."
                elif not expected:
                    entry["feedback"] = "No expected answer was configured for this question."

                possible += 1.0
                total += entry["score"]
            elif qtype == "multi_select":
                entry["score"], entry["feedback"] = _evaluate_multi_select(selected_answers, correct_options)
                possible += 1.0
                total += entry["score"]
            elif qtype == "true_false":
                expected = correct
                if selected_answers and _normalize_text(selected_answers[0]) == _normalize_text(expected):
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

    return render_template(
        "quiz.html",
        lecture_id=lecture_id,
        questions=questions,
        results=results,
        total_score=total_score,
        lecture_meta=load_lecture_metadata(lecture_id),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
