import json
import os
import unittest

try:
    import flask  # noqa: F401
except ModuleNotFoundError:
    HAS_FLASK = False
else:
    HAS_FLASK = True

try:
    from app import create_app
    from app.config.settings import TRANSCRIPT_DIR
    from app.core.coaching_engine import record_quiz_attempt
    from app.core.review_engine import apply_review_rating, build_lecture_review_summary, sync_review_progress
    from app.utils.study_storage import (
        coaching_path,
        flashcard_deck_path,
        lecture_meta_path,
        load_lecture_metadata,
        quiz_attempts_path,
        review_progress_path,
        save_flashcard_deck,
        save_lecture_metadata,
    )
except ModuleNotFoundError as exc:
    create_app = None
    if exc.name != "flask":
        raise


@unittest.skipIf(not HAS_FLASK or create_app is None, "Flask is not installed in this environment.")
class StudyFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def tearDown(self):
        lecture_ids = [
            "test_review_lecture",
            "test_flashcard_route",
            "test_coaching_lecture",
        ]
        for lecture_id in lecture_ids:
            for path in (
                lecture_meta_path(lecture_id),
                flashcard_deck_path(lecture_id),
                review_progress_path(lecture_id),
                quiz_attempts_path(lecture_id),
                coaching_path(lecture_id),
                os.path.join(TRANSCRIPT_DIR, f"{lecture_id}_chunks.json"),
            ):
                if os.path.exists(path):
                    os.remove(path)

    def test_flashcards_route_renders_lecture_block_from_saved_deck(self):
        lecture_id = "test_flashcard_route"
        save_lecture_metadata(
            lecture_id,
            {
                "lecture_id": lecture_id,
                "title": "Computer Networks",
                "source_type": "upload",
                "source_label": "computer_networks.mp4",
                "source_url": "",
            },
        )
        deck = {
            "lecture_id": lecture_id,
            "version": 1,
            "generated_at": "2026-03-18T00:00:00Z",
            "card_count": 1,
            "cards": [
                {
                    "card_id": "card_a",
                    "lecture_id": lecture_id,
                    "front": "What is flow control?",
                    "back": "It prevents a fast sender from overwhelming a slow receiver.",
                    "hint": "Think sender versus receiver speed.",
                    "concept": "Flow Control",
                    "source_segments": [{"start": 12.0, "end": 19.0, "text": "Flow control keeps sender and receiver balanced."}],
                }
            ],
        }
        save_flashcard_deck(lecture_id, deck)
        sync_review_progress(lecture_id, deck["cards"])

        response = self.client.get(f"/flashcards?lecture_id={lecture_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Computer Networks", response.data)
        self.assertIn(b"Reveal Answer", response.data)

    def test_review_rating_clears_due_card_for_now(self):
        lecture_id = "test_review_lecture"
        save_lecture_metadata(
            lecture_id,
            {
                "lecture_id": lecture_id,
                "title": "Operating Systems",
                "source_type": "upload",
                "source_label": "os.mp4",
            },
        )
        deck_cards = [
            {
                "card_id": "card_os",
                "lecture_id": lecture_id,
                "front": "What is a semaphore?",
                "back": "A synchronization primitive used to control access to shared resources.",
                "hint": "Think process coordination.",
                "concept": "Semaphore",
                "source_segments": [],
            }
        ]
        save_flashcard_deck(
            lecture_id,
            {
                "lecture_id": lecture_id,
                "version": 1,
                "generated_at": "2026-03-18T00:00:00Z",
                "card_count": 1,
                "cards": deck_cards,
            },
        )
        sync_review_progress(lecture_id, deck_cards)

        apply_review_rating(lecture_id, "card_os", "good")
        summary = build_lecture_review_summary(lecture_id)

        self.assertEqual(summary["due_today_count"], 0)
        self.assertEqual(summary["overdue_count"], 0)
        self.assertTrue(summary["completed_for_now"])

    def test_record_quiz_attempt_builds_coaching_payload(self):
        lecture_id = "test_coaching_lecture"
        save_lecture_metadata(
            lecture_id,
            {
                "lecture_id": lecture_id,
                "title": "Transport Layer",
                "source_type": "upload",
                "source_label": "transport.mp4",
            },
        )
        with open(os.path.join(TRANSCRIPT_DIR, f"{lecture_id}_chunks.json"), "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "start": 0.0,
                        "end": 18.0,
                        "text": "Flow control prevents a sender from overwhelming a receiver in the transport layer.",
                    },
                    {
                        "start": 18.0,
                        "end": 40.0,
                        "text": "Congestion control keeps the network from becoming overloaded.",
                    },
                ],
                handle,
                ensure_ascii=False,
                indent=2,
            )

        attempt = record_quiz_attempt(
            lecture_id,
            [
                {
                    "question": "How does flow control help the receiver?",
                    "type": "short_answer",
                    "user_answer": "It manages packets poorly.",
                    "correct": "It prevents the sender from overwhelming the receiver.",
                    "assessment": "Incorrect",
                    "feedback": "You missed the receiver protection idea.",
                    "explanation": "Flow control protects the receiver side of the transport layer.",
                    "score": 0.0,
                }
            ],
            40.0,
        )

        metadata = load_lecture_metadata(lecture_id)
        self.assertEqual(metadata.get("last_quiz_score"), 40.0)
        self.assertTrue(attempt["coaching"]["weak_concepts"])
        self.assertEqual(attempt["coaching"]["weak_concepts"][0]["label"], "Flow Control")


if __name__ == "__main__":
    unittest.main()
