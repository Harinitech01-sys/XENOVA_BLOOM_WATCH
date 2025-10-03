import unittest
from app import app

class FrontendViewsTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)

    def test_dashboard(self):
        resp = self.client.get('/dashboard')
        self.assertEqual(resp.status_code, 200)

if __name__ == "__main__":
    unittest.main()
