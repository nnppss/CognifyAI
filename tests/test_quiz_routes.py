import unittest

from app.api.quiz_utils import evaluate_multi_select, prepare_quiz_questions


class QuizRouteHelpersTests(unittest.TestCase):
    def test_prepare_quiz_questions_normalizes_multi_select_answers(self):
        prepared = prepare_quiz_questions(
            [
                {
                    "type": "multiple_correct",
                    "question": "Which protocols are transport-layer protocols?",
                    "options": ["TCP", "UDP", "HTTP"],
                    "correct": "A and B",
                }
            ]
        )

        self.assertEqual(prepared[0]["type"], "multi_select")
        self.assertEqual(prepared[0]["resolved_correct_options"], ["TCP", "UDP"])

    def test_prepare_quiz_questions_promotes_multi_answer_mcq(self):
        prepared = prepare_quiz_questions(
            [
                {
                    "type": "mcq",
                    "question": "Pick every valid answer.",
                    "options": ["Alpha", "Beta", "Gamma"],
                    "correct": ["Alpha", "Gamma"],
                }
            ]
        )

        self.assertEqual(prepared[0]["type"], "multi_select")
        self.assertEqual(prepared[0]["resolved_correct_options"], ["Alpha", "Gamma"])

    def test_evaluate_multi_select_returns_partial_credit_for_subset(self):
        score, feedback = evaluate_multi_select(["TCP"], ["TCP", "UDP"])

        self.assertEqual(score, 0.5)
        self.assertEqual(feedback, "Missing: UDP.")

    def test_evaluate_multi_select_reports_wrong_answers(self):
        score, feedback = evaluate_multi_select(["TCP", "HTTP"], ["TCP", "UDP"])

        self.assertEqual(score, 0.0)
        self.assertEqual(feedback, "Not correct: HTTP. Missing: UDP.")


if __name__ == "__main__":
    unittest.main()
