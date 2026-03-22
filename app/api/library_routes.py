import os

from flask import Blueprint, flash, redirect, render_template, url_for

from app.config.settings import (
    COACHING_DIR,
    FLASHCARD_DIR,
    INDEX_DIR,
    LECTURE_META_DIR,
    QUIZ_ATTEMPT_DIR,
    QUIZ_DIR,
    REVIEW_DIR,
    SUMMARY_DIR,
    TRANSCRIPT_DIR,
)
from app.core.library_engine import build_library_dashboard

library_bp = Blueprint("library", __name__)


@library_bp.route("/lectures")
def dashboard():
    payload = build_library_dashboard()
    return render_template("library.html", dashboard=payload)


@library_bp.route("/lectures/<lecture_id>/delete", methods=["POST"])
def delete_lecture(lecture_id: str):
    lecture_id = lecture_id.strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("library.dashboard"))

    file_paths = [
        os.path.join(TRANSCRIPT_DIR, f"{lecture_id}_chunks.json"),
        os.path.join(INDEX_DIR, f"{lecture_id}_embeddings.npy"),
        os.path.join(INDEX_DIR, f"{lecture_id}_segments.json"),
        os.path.join(SUMMARY_DIR, f"{lecture_id}_summary.json"),
        os.path.join(SUMMARY_DIR, f"{lecture_id}_notes.json"),
        os.path.join(QUIZ_DIR, f"{lecture_id}_quiz.json"),
        os.path.join(FLASHCARD_DIR, f"{lecture_id}_flashcards.json"),
        os.path.join(REVIEW_DIR, f"{lecture_id}_review.json"),
        os.path.join(QUIZ_ATTEMPT_DIR, f"{lecture_id}_attempts.json"),
        os.path.join(COACHING_DIR, f"{lecture_id}_coaching.json"),
        os.path.join(LECTURE_META_DIR, f"{lecture_id}.json"),
    ]

    removed = 0
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass

    flash(f"Lecture '{lecture_id}' deleted ({removed} files removed).", "success")
    return redirect(url_for("library.dashboard"))

