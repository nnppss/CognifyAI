from flask import Blueprint, flash, redirect, render_template, request, url_for

from app.api.helpers import build_qa_redirect, load_lecture_metadata, mark_lecture_opened
from app.core.library_engine import build_due_groups, build_library_dashboard

flashcards_bp = Blueprint("flashcards", __name__)


@flashcards_bp.route("/flashcards")
def flashcards_view():
    lecture_id = (request.args.get("lecture_id") or "").strip()

    if not lecture_id:
        dashboard = build_library_dashboard()
        return render_template(
            "flashcards.html",
            dashboard=dashboard,
            due_groups=build_due_groups(dashboard),
            lecture_id="",
            lecture_meta={},
            lecture_summary={},
            current_card=None,
            queue_cards=[],
            coaching={},
        )

    lecture_meta = load_lecture_metadata(lecture_id)
    if not lecture_meta:
        flash("Lecture not found. Process a lecture first.", "error")
        return redirect(url_for("lectures.index"))

    try:
        from app.core.coaching_engine import build_coaching_payload
        from app.core.flashcard_engine import ensure_flashcard_deck
        from app.core.review_engine import build_lecture_review_summary, get_due_cards_for_lecture

        mark_lecture_opened(lecture_id)
        deck = ensure_flashcard_deck(lecture_id)
        lecture_summary = build_lecture_review_summary(lecture_id)
        queue_cards = get_due_cards_for_lecture(lecture_id)
        current_card = queue_cards[0] if queue_cards else None
        coaching = build_coaching_payload(lecture_id)
    except Exception as exc:
        flash(f"Flashcard view failed: {exc}", "error")
        return build_qa_redirect(lecture_id)

    dashboard = build_library_dashboard()
    return render_template(
        "flashcards.html",
        dashboard=dashboard,
        due_groups=build_due_groups(dashboard),
        lecture_id=lecture_id,
        lecture_meta=lecture_meta,
        lecture_summary={**lecture_summary, "card_count": deck.get("card_count", 0)},
        current_card=current_card,
        queue_cards=queue_cards[1:6] if current_card else [],
        coaching=coaching,
    )


@flashcards_bp.route("/flashcards/review", methods=["POST"])
def review_flashcard():
    lecture_id = (request.form.get("lecture_id") or "").strip()
    card_id = (request.form.get("card_id") or "").strip()
    rating = (request.form.get("rating") or "").strip().lower()

    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("lectures.index"))
    if not card_id:
        flash("Card ID is required.", "error")
        return redirect(url_for("flashcards.flashcards_view", lecture_id=lecture_id))

    try:
        from app.core.review_engine import apply_review_rating, get_due_cards_for_lecture

        apply_review_rating(lecture_id, card_id, rating)
        remaining = get_due_cards_for_lecture(lecture_id)
    except Exception as exc:
        flash(f"Flashcard review failed: {exc}", "error")
        return redirect(url_for("flashcards.flashcards_view", lecture_id=lecture_id))

    if not remaining:
        flash("Lecture block completed. Back to your library dashboard.", "success")
        return redirect(url_for("library.dashboard"))

    return redirect(url_for("flashcards.flashcards_view", lecture_id=lecture_id))


@flashcards_bp.route("/flashcards/regenerate", methods=["POST"])
def regenerate_flashcards():
    lecture_id = (request.form.get("lecture_id") or "").strip()
    if not lecture_id:
        flash("Lecture ID is required.", "error")
        return redirect(url_for("lectures.index"))

    try:
        from app.core.flashcard_engine import ensure_flashcard_deck

        ensure_flashcard_deck(lecture_id, force=True)
        flash("Flashcards regenerated successfully.", "success")
    except Exception as exc:
        flash(f"Flashcard generation failed: {exc}", "error")
        return build_qa_redirect(lecture_id)

    return redirect(url_for("flashcards.flashcards_view", lecture_id=lecture_id))
