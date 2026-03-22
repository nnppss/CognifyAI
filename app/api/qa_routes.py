import os

from flask import Blueprint, Response, flash, redirect, render_template, request, stream_with_context, url_for

from app.api.helpers import build_qa_redirect, chunks_path, load_lecture_metadata, mark_lecture_opened, parse_top_k, sse_event

qa_bp = Blueprint("qa", __name__)


@qa_bp.route("/qa", methods=["GET", "POST"])
def qa_view():
    lecture_id = (request.args.get("lecture_id") or "").strip()
    src_url = (request.args.get("src_url") or "").strip()

    answer = None
    segments = []
    timestamp = None
    mode = None
    question = ""
    top_k = 3

    if request.method == "POST":
        from app.core.qa_engine import LectureQA

        lecture_id = (request.form.get("lecture_id") or "").strip()
        src_url = (request.form.get("src_url") or "").strip()
        question = (request.form.get("question") or "").strip()
        top_k = parse_top_k(request.form.get("top_k"))

        if not lecture_id:
            flash("Lecture ID is required.", "error")
            return redirect(url_for("qa.qa_view"))
        if not question:
            flash("Please enter a question.", "error")
            return build_qa_redirect(lecture_id)

        try:
            engine = LectureQA(lecture_id)
            result = engine.answer_question(question, top_k=top_k)
            answer = result["answer"]
            segments = result["segments"]
            timestamp = int(result["timestamp"])
            mode = result.get("mode", "rag")
        except Exception as exc:
            flash(f"Error during question answering: {exc}", "error")
            return build_qa_redirect(lecture_id)

    has_chunks = bool(lecture_id) and os.path.exists(chunks_path(lecture_id))
    lecture_meta = mark_lecture_opened(lecture_id) if lecture_id else {}
    if lecture_id and not lecture_meta:
        lecture_meta = load_lecture_metadata(lecture_id)

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


@qa_bp.route("/qa/stream", methods=["POST"])
def qa_stream():
    lecture_id = (request.form.get("lecture_id") or "").strip()
    question = (request.form.get("question") or "").strip()
    top_k = parse_top_k(request.form.get("top_k"))

    if not lecture_id:
        return {"error": "Lecture ID is required."}, 400
    if not question:
        return {"error": "Please enter a question."}, 400

    def generate():
        try:
            from app.core.qa_engine import LectureQA

            engine = LectureQA(lecture_id)
            for event in engine.answer_question_stream(question, top_k=top_k):
                event_name = event.get("event", "message")
                payload = event.get("data", {})
                yield sse_event(event_name, payload)
        except Exception as exc:
            yield sse_event("error", {"message": f"Error during question answering: {exc}"})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers=headers,
    )
