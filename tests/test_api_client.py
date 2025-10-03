import unittest
from api_client import get_dashboard_data

class TestAPIClient(unittest.TestCase):
    def test_get_dashboard_data(self):
        data = get_dashboard_data()
        self.assertIsInstance(data, dict)

if __name__ == "__main__":
    unittest.main()
