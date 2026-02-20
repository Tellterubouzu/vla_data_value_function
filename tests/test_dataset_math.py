import unittest

from src.data.airoa_moma_vf_dataset import (
    compute_return,
    compute_value_and_bin,
    map_step_to_frame_index,
    compute_segment_range,
)


class DatasetMathTest(unittest.TestCase):
    def test_compute_return_success(self):
        self.assertEqual(compute_return(length=11, timestep=0, success=True, c_fail=20), -10)
        self.assertEqual(compute_return(length=11, timestep=10, success=True, c_fail=20), 0)

    def test_compute_return_failure(self):
        self.assertEqual(compute_return(length=11, timestep=10, success=False, c_fail=20), -20)
        self.assertEqual(compute_return(length=11, timestep=5, success=False, c_fail=20), -25)

    def test_compute_value_and_bin(self):
        v_success, b_success = compute_value_and_bin(
            length=11,
            timestep=0,
            success=True,
            l_max_task=20,
            num_bins=201,
        )
        self.assertAlmostEqual(v_success, -0.5)
        self.assertEqual(b_success, 100)

        v_terminal, b_terminal = compute_value_and_bin(
            length=11,
            timestep=10,
            success=True,
            l_max_task=20,
            num_bins=201,
        )
        self.assertAlmostEqual(v_terminal, 0.0)
        self.assertEqual(b_terminal, 200)

        v_fail, b_fail = compute_value_and_bin(
            length=11,
            timestep=5,
            success=False,
            l_max_task=20,
            num_bins=201,
        )
        self.assertAlmostEqual(v_fail, -1.0)
        self.assertEqual(b_fail, 0)

    def test_map_step_to_frame_index(self):
        self.assertEqual(map_step_to_frame_index(timestep=0, length=10, n_video_frames=5), 0)
        self.assertEqual(map_step_to_frame_index(timestep=9, length=10, n_video_frames=5), 4)
        self.assertEqual(map_step_to_frame_index(timestep=5, length=10, n_video_frames=5), 2)
        self.assertEqual(map_step_to_frame_index(timestep=0, length=1, n_video_frames=50), 0)

    def test_compute_segment_range(self):
        start, seg_len = compute_segment_range(
            total_frames=1000,
            from_ts=1.0,
            to_ts=2.0,
            fps=30.0,
        )
        self.assertEqual(start, 30)
        self.assertEqual(seg_len, 30)

        start2, seg_len2 = compute_segment_range(
            total_frames=100,
            from_ts=None,
            to_ts=None,
            fps=30.0,
        )
        self.assertEqual(start2, 0)
        self.assertEqual(seg_len2, 100)


if __name__ == "__main__":
    unittest.main()
