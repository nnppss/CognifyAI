import glob
import os
import re
import tempfile
import json
from typing import List, Dict

from config import TRANSCRIPT_DIR, MAX_CHUNK_WORDS


# --------- Video ID extraction ---------

def get_video_id(url: str) -> str:
    """
    Extract a valid 11-character YouTube video ID from many possible URL formats.
    Raises ValueError if a valid ID cannot be found.
    """
    url = (url or "").strip()

    # If the user already pasted a raw ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
        r"/live/([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)

    raise ValueError(
        "Could not extract a valid YouTube video ID. "
        "Please paste a real YouTube lecture link (with the CC icon) or a raw 11-character ID."
    )


# --------- Transcript fetching ---------

def _parse_vtt_timestamp(ts: str) -> float:
    """Convert WebVTT timestamp 'HH:MM:SS.mmm' to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s_ms = parts
    elif len(parts) == 2:
        h, m, s_ms = 0, parts[0], parts[1]
    else:
        raise ValueError(f"Bad VTT timestamp: {ts}")

    h = int(h)
    m = int(m)
    if "." in s_ms:
        s, ms = s_ms.split(".")
        s = int(s)
        ms = int(ms[:3])
    else:
        s = int(s_ms)
        ms = 0
    return h * 3600 + m * 60 + s + ms / 1000.0


def _fetch_transcript_via_ytdlp(video_id: str) -> List[Dict]:
    """
    Use yt-dlp to fetch English subtitles/auto-captions (.vtt) and parse them.
    Returns list of {text, start, duration}.
    """
    import yt_dlp
    from webvtt import WebVTT

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, f"{video_id}.%(ext)s")
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB", "en-IN"],
            "subtitlesformat": "vtt",
            "outtmpl": outtmpl,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        vtt_files = glob.glob(os.path.join(tmpdir, f"{video_id}*.vtt"))
        if not vtt_files:
            raise RuntimeError("yt-dlp did not download any .vtt subtitle file.")

        vtt_path = vtt_files[0]

        segments: List[Dict] = []
        for caption in WebVTT().read(vtt_path):
            text = caption.text.replace("\n", " ").strip()
            if not text:
                continue
            start = _parse_vtt_timestamp(caption.start)
            end = _parse_vtt_timestamp(caption.end)
            segments.append(
                {
                    "text": text,
                    "start": float(start),
                    "duration": float(end - start),
                }
            )

        if not segments:
            raise RuntimeError("Parsed VTT contained no usable caption segments.")

        return segments


def fetch_transcript(video_id: str) -> List[Dict]:
    """
    Robustly fetch transcript segments for a YouTube video.

    - Try youtube-transcript-api first (manual or auto English captions).
    - If that fails, fall back to yt-dlp + WebVTT parsing.
    """
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
    except Exception as e:
        raise RuntimeError(f"youtube-transcript-api is not available: {e}")

    try:
        # Try standard English variants
        return YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-GB", "en-IN"]
        )
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        # Fall through to yt-dlp
        pass
    except Exception:
        # Unknown error, still try yt-dlp
        pass

    # yt-dlp fallback
    try:
        return _fetch_transcript_via_ytdlp(video_id)
    except Exception as e2:
        raise RuntimeError(
            f"No usable transcript found for this video (captions disabled or not in English). "
            f"Details: {e2}"
        )


# --------- Chunking & saving ---------

def merge_segments(raw_transcript: List[Dict], max_words: int = MAX_CHUNK_WORDS) -> List[Dict]:
    """
    Merge small caption entries into chunks of roughly max_words words.
    Each output chunk has: start, end, text.
    """
    chunks: List[Dict] = []
    cur_words: List[str] = []
    cur_start = None
    cur_end = None

    for entry in raw_transcript:
        text = (entry.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue

        words = text.split()
        if cur_start is None:
            cur_start = float(entry.get("start", 0.0))

        # compute end from duration or 'end'
        if "duration" in entry:
            cur_end = float(entry["start"]) + float(entry.get("duration", 0.0))
        else:
            cur_end = float(entry.get("end", entry.get("start", 0.0)))

        cur_words.extend(words)

        if len(cur_words) >= max_words:
            chunks.append(
                {"start": float(cur_start), "end": float(cur_end), "text": " ".join(cur_words)}
            )
            cur_words, cur_start, cur_end = [], None, None

    if cur_words:
        chunks.append(
            {"start": float(cur_start), "end": float(cur_end), "text": " ".join(cur_words)}
        )

    return chunks


def save_chunks(lecture_id: str, chunks: List[Dict]) -> str:
    """Save chunks to data/transcripts/{lecture_id}_chunks.json"""
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    path = os.path.join(TRANSCRIPT_DIR, f"{lecture_id}_chunks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return path
