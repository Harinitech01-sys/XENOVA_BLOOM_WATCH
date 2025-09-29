import unittest
from ndvi_processing import calculate_ndvi

class TestNDVIProcessing(unittest.TestCase):
    def test_ndvi(self):
        data = {"NIR": [0.8, 0.7], "RED": [0.4, 0.3]}
        ndvi = calculate_ndvi(data)
        self.assertAlmostEqual(ndvi[0], 0.333, places=2)

if __name__ == "__main__":
    unittest.main()
