from typing import Any, Dict, List

from app.core.review_engine import build_lecture_review_summary
from app.utils.study_storage import (
    flashcard_deck_path,
    list_lecture_metadata,
    notes_cache_path,
    parse_iso_datetime,
    quiz_cache_path,
    summary_cache_path,
    update_lecture_metadata,
)


def _sort_timestamp(raw_value: str | None) -> float:
    parsed = parse_iso_datetime(raw_value)
    if parsed is None:
        return 0.0
    return parsed.timestamp()


def _lecture_asset_flags(lecture_id: str) -> Dict[str, bool]:
    import os

    return {
        "has_summary": os.path.exists(summary_cache_path(lecture_id)),
        "has_notes": os.path.exists(notes_cache_path(lecture_id)),
        "has_quiz": os.path.exists(quiz_cache_path(lecture_id)),
        "has_flashcards": os.path.exists(flashcard_deck_path(lecture_id)),
    }


def build_library_dashboard() -> Dict[str, Any]:
    lectures = []
    total_due_today = 0
    total_overdue = 0
    lectures_in_rotation = 0

    for lecture_meta in list_lecture_metadata():
        lecture_id = str(lecture_meta.get("lecture_id") or "").strip()
        if not lecture_id:
            continue

        assets = _lecture_asset_flags(lecture_id)
        review = build_lecture_review_summary(lecture_id)
        lecture = {
            **lecture_meta,
            **assets,
            **review,
            "weak_concepts": lecture_meta.get("weak_concepts", []),
        }
        lecture["review_total"] = lecture["due_today_count"] + lecture["overdue_count"]
        if lecture["review_total"] > 0:
            lectures_in_rotation += 1

        total_due_today += lecture["due_today_count"]
        total_overdue += lecture["overdue_count"]

        update_lecture_metadata(
            lecture_id,
            has_summary=lecture["has_summary"],
            has_notes=lecture["has_notes"],
            has_quiz=lecture["has_quiz"],
            has_flashcards=lecture["has_flashcards"],
            due_today_count=lecture["due_today_count"],
            overdue_count=lecture["overdue_count"],
        )
        lectures.append(lecture)

    lectures.sort(
        key=lambda item: (
            -int(item.get("overdue_count") or 0),
            -int(item.get("due_today_count") or 0),
            -_sort_timestamp(item.get("last_opened_at")),
            str(item.get("title") or item.get("lecture_id") or "").lower(),
        )
    )

    weak_lectures = [lecture for lecture in lectures if lecture.get("weak_concepts")]
    study_next = weak_lectures[:3] if weak_lectures else lectures[:3]

    return {
        "stats": {
            "due_today": total_due_today,
            "overdue": total_overdue,
            "lectures_in_rotation": lectures_in_rotation,
            "lecture_count": len(lectures),
        },
        "lectures": lectures,
        "study_next": study_next,
        "has_due_cards": any((lecture.get("review_total") or 0) > 0 for lecture in lectures),
    }


def build_due_groups(dashboard: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    if dashboard is None:
        dashboard = build_library_dashboard()
    due_groups = [lecture for lecture in dashboard["lectures"] if (lecture.get("review_total") or 0) > 0]
    due_groups.sort(
        key=lambda item: (
            -int(item.get("overdue_count") or 0),
            -int(item.get("due_today_count") or 0),
            -_sort_timestamp(item.get("last_opened_at")),
        )
    )
    return due_groups
