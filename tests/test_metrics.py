import unittest

import numpy as np

from src.utils.metrics import bin_centers, mean_absolute_error


class MetricsTest(unittest.TestCase):
    def test_bin_centers(self):
        centers = bin_centers(num_bins=201)
        self.assertEqual(len(centers), 201)
        self.assertAlmostEqual(float(centers[0]), -1.0, places=6)
        self.assertAlmostEqual(float(centers[-1]), 0.0, places=6)

    def test_mae_numpy_path(self):
        pred = np.array([-1.0, -0.5, 0.0], dtype=np.float32)
        gt = np.array([-1.0, -0.25, -0.25], dtype=np.float32)
        mae = mean_absolute_error(pred, gt)
        self.assertAlmostEqual(mae, (0.0 + 0.25 + 0.25) / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
