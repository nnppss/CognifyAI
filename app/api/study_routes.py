from flask import Blueprint, flash, redirect, render_template, request, send_file, url_for

from app.api.helpers import build_qa_redirect, flash_generation_failure, load_chunks, load_lecture_metadata, mark_lecture_opened

study_bp = Blueprint("study", __name__)


@study_bp.route("/summary")
def summary_view():
    lecture_id = (request.args.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("lectures.index"))

    try:
        from app.core.summary_engine import generate_summary

        chunks = load_chunks(lecture_id)
        payload = generate_summary(lecture_id, chunks)
    except Exception as exc:
        flash_generation_failure("Summary generation", exc)
        return build_qa_redirect(lecture_id)

    return render_template(
        "summary.html",
        lecture_id=lecture_id,
        title="Lecture Summary",
        content=payload.get("summary", ""),
        mode="summary",
        lecture_meta=mark_lecture_opened(lecture_id) or load_lecture_metadata(lecture_id),
    )


@study_bp.route("/notes")
def notes_view():
    lecture_id = (request.args.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("lectures.index"))

    try:
        from app.core.summary_engine import generate_detailed_notes

        chunks = load_chunks(lecture_id)
        payload = generate_detailed_notes(lecture_id, chunks)
    except Exception as exc:
        flash_generation_failure("Notes generation", exc)
        return build_qa_redirect(lecture_id)

    return render_template(
        "summary.html",
        lecture_id=lecture_id,
        title="Detailed Notes",
        content=payload.get("notes", ""),
        mode="notes",
        lecture_meta=mark_lecture_opened(lecture_id) or load_lecture_metadata(lecture_id),
    )


@study_bp.route("/download/<doc_type>")
def download_doc(doc_type: str):
    lecture_id = (request.args.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("lectures.index"))
    if doc_type not in {"summary", "notes"}:
        flash("Invalid document type.", "error")
        return build_qa_redirect(lecture_id)

    try:
        from app.core.summary_engine import generate_detailed_notes, generate_summary
        from app.utils.pdf_generator import generate_pdf

        chunks = load_chunks(lecture_id)
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
        return build_qa_redirect(lecture_id)
