import unittest
from data_acquisition import fetch_satellite_data

class TestDataAcquisition(unittest.TestCase):
    def test_fetch_data(self):
        result = fetch_satellite_data("India")
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
