from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, flash, redirect, render_template, request, url_for

from app.api.helpers import build_qa_redirect, load_chunks, load_lecture_metadata, reuse_processed_lecture

lectures_bp = Blueprint("lectures", __name__)


@lectures_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        from app.core.indexing import build_index
        from app.services.frame_ocr_utils import extract_frame_ocr_segments
        from app.services.speechmatics_transcribe import transcribe_audio_speechmatics
        from app.services.youtube_transcript_utils import fetch_youtube_transcript
        from app.utils.media_utils import (
            allowed_file,
            create_upload_lecture_id,
            download_youtube_audio,
            download_youtube_video_for_frames,
            extract_audio_ffmpeg,
            save_lecture_metadata,
            save_uploaded_video,
        )
        from app.utils.transcript_utils import get_video_id, merge_segments, save_chunks

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

                if reuse_processed_lecture(lecture_id):
                    if not load_lecture_metadata(lecture_id):
                        chunks = load_chunks(lecture_id)
                        save_lecture_metadata(
                            lecture_id,
                            {
                                "lecture_id": lecture_id,
                                "source_type": source_type,
                                "source_label": source_label,
                                "source_url": source_url,
                                "transcript_source": "cached",
                                "detected_language": "",
                                "ocr_segment_count": 0,
                                "ocr_error": "",
                                "chunk_count": len(chunks),
                            },
                        )
                    flash(f"Lecture already processed. Reusing saved materials for {lecture_id}.", "success")
                    return build_qa_redirect(lecture_id)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    frame_future = executor.submit(download_youtube_video_for_frames, youtube_url, lecture_id)
                    transcript_future = executor.submit(fetch_youtube_transcript, video_id)

                    try:
                        raw_segments, transcript_language = transcript_future.result()
                        detected_language = transcript_language or ""
                        transcript_source = "youtube-transcript"
                    except Exception:
                        audio_path = download_youtube_audio(youtube_url, lecture_id)
                        raw_segments = transcribe_audio_speechmatics(audio_path, language="auto")
                        transcript_source = "speechmatics-youtube-fallback"
                        detected_language = "auto"

                    try:
                        video_path = frame_future.result()
                    except Exception as exc:
                        video_path = ""
                        ocr_error = f"Frame analysis skipped: {exc}"

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
            return redirect(url_for("lectures.index"))

        flash(f"Lecture processed successfully: {lecture_id}, chunks={len(chunks)}", "success")
        return build_qa_redirect(lecture_id)

    return render_template("index.html")
