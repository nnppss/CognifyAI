import unittest

try:
    import flask  # noqa: F401
except ModuleNotFoundError:
    HAS_FLASK = False
else:
    HAS_FLASK = True

try:
    from app import create_app
except ModuleNotFoundError as exc:
    create_app = None
    if exc.name != "flask":
        raise


@unittest.skipIf(not HAS_FLASK or create_app is None, "Flask is not installed in this environment.")
class AppFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_registers_feature_blueprints(self):
        self.assertIn("library", self.app.blueprints)
        self.assertIn("flashcards", self.app.blueprints)
        self.assertIn("lectures", self.app.blueprints)
        self.assertIn("qa", self.app.blueprints)
        self.assertIn("study", self.app.blueprints)
        self.assertIn("quiz", self.app.blueprints)

    def test_index_page_renders(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CognifyAI", response.data)

    def test_summary_redirects_without_lecture_id(self):
        response = self.client.get("/summary")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/"))

    def test_quiz_redirects_without_lecture_id(self):
        response = self.client.get("/quiz")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/"))

    def test_stream_endpoint_validates_lecture_id(self):
        response = self.client.post("/qa/stream", data={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Lecture ID is required."})

    def test_library_dashboard_renders(self):
        response = self.client.get("/lectures")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Library Dashboard", response.data)

    def test_global_flashcards_view_renders(self):
        response = self.client.get("/flashcards")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Due Today Review", response.data)


if __name__ == "__main__":
    unittest.main()
