import os
import sys

# Make sure local modules can be imported when running this file directly
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, redirect, url_for, flash

from transcript_utils import (
    get_video_id,
    fetch_transcript,
    merge_segments,
    save_chunks,
)
from indexing import build_index
from qa_engine import LectureQA
from config import TRANSCRIPT_DIR

app = Flask(__name__)
app.secret_key = "cognifyai_sem7_secret"  # for flash messages


def yt_ts_link(url: str, seconds: int) -> str | None:
    """
    Build a YouTube URL that jumps to the given timestamp in seconds.
    """
    if not url:
        return None
    base = url.split("&")[0]
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}t={int(seconds)}s"


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Step 1 page: process a YouTube lecture.

    - User enters a YouTube URL.
    - We extract the video ID, fetch transcript, chunk it, build index.
    - On success, redirect to /qa with lecture_id and src_url.
    """
    if request.method == "POST":
        url = (request.form.get("youtube_url") or "").strip()
        placeholder = "https://www.youtube.com/watch?v=..."

        if not url or url == placeholder:
            flash("Please paste an actual YouTube lecture URL (not the placeholder).", "error")
            return redirect(url_for("index"))

        # 1) Extract video ID
        try:
            video_id = get_video_id(url)
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("index"))

        # 2) Fetch transcript
        try:
            raw = fetch_transcript(video_id)
        except Exception as e:
            flash(f"Could not fetch transcript: {e}", "error")
            return redirect(url_for("index"))

        # 3) Chunk + save + build index
        try:
            chunks = merge_segments(raw)
            save_chunks(video_id, chunks)
            build_index(video_id)
        except Exception as e:
            flash(f"Failed to build semantic index: {e}", "error")
            return redirect(url_for("index"))

        flash(
            f"Lecture processed successfully: ID {video_id}, chunks={len(chunks)}",
            "success",
        )
        # Redirect to Q&A for this lecture
        return redirect(url_for("qa", lecture_id=video_id, src_url=url))

    # GET
    return render_template("index.html")


@app.route("/qa", methods=["GET", "POST"])
def qa():
    """
    Step 2 page: ask questions about a processed lecture.
    """
    # Defaults from query string
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
            return redirect(url_for("qa", lecture_id=lecture_id, src_url=src_url))

        try:
            engine = LectureQA(lecture_id)
            result = engine.answer_question(question, top_k=top_k)
            answer = result["answer"]
            segments = result["segments"]
            timestamp = int(result["timestamp"])
            mode = result.get("mode", "rag")
        except Exception as e:
            flash(f"Error during question answering: {e}", "error")
            return redirect(url_for("qa", lecture_id=lecture_id, src_url=src_url))

    # Check whether lecture has processed chunks
    has_chunks = False
    if lecture_id:
        chunks_file = os.path.join(TRANSCRIPT_DIR, f"{lecture_id}_chunks.json")
        has_chunks = os.path.exists(chunks_file)

    yt_link = yt_ts_link(src_url, timestamp) if (src_url and timestamp is not None) else None

    return render_template(
        "qa.html",
        lecture_id=lecture_id,
        src_url=src_url,
        question=question,
        answer=answer,
        segments=segments,
        timestamp=timestamp,
        yt_link=yt_link,
        mode=mode,
        has_chunks=has_chunks,
        top_k=top_k,
    )


if __name__ == "__main__":
    # Debug=True is fine for viva/demo; not for real production.
    app.run(host="127.0.0.1", port=8000, debug=True)
