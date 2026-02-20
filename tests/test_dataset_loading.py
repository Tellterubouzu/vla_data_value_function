import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.data.airoa_moma_vf_dataset import (
    AiroaMomaEpisodeStore,
    AiroaMomaValueFrameDataset,
    compute_l_max_by_task,
)


class DummyVideoCache:
    def get_num_frames(self, video_path: Path) -> int:
        return 100

    def read_frame(self, video_path: Path, frame_index: int) -> Image.Image:
        c = int(frame_index % 255)
        return Image.new("RGB", (8, 8), (c, c, c))


class DatasetLoadingTest(unittest.TestCase):
    def _write_mock_dataset(self, root: Path) -> None:
        (root / "videos" / "observation.image.head" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (root / "videos" / "observation.image.hand" / "chunk-000").mkdir(parents=True, exist_ok=True)

        # Empty files are sufficient because tests mock video decoding.
        for idx in [0, 1, 2]:
            (root / "videos" / "observation.image.head" / "chunk-000" / f"file-{idx:03d}.mp4").touch()
            (root / "videos" / "observation.image.hand" / "chunk-000" / f"file-{idx:03d}.mp4").touch()

        rows = [
            {
                "episode_index": 0,
                "length": 10,
                "task_success": True,
                "short_horizon_task": "task_a",
                "videos/observation.image.head/chunk_index": 0,
                "videos/observation.image.head/file_index": 0,
                "videos/observation.image.head/from_timestamp": 0.0,
                "videos/observation.image.head/to_timestamp": 10.0 / 30.0,
                "videos/observation.image.hand/chunk_index": 0,
                "videos/observation.image.hand/file_index": 0,
                "videos/observation.image.hand/from_timestamp": 0.0,
                "videos/observation.image.hand/to_timestamp": 10.0 / 30.0,
            },
            {
                "episode_index": 1,
                "length": 20,
                "task_success": False,
                "short_horizon_task": "task_a",
                "videos/observation.image.head/chunk_index": 0,
                "videos/observation.image.head/file_index": 1,
                "videos/observation.image.head/from_timestamp": 0.0,
                "videos/observation.image.head/to_timestamp": 20.0 / 30.0,
                "videos/observation.image.hand/chunk_index": 0,
                "videos/observation.image.hand/file_index": 1,
                "videos/observation.image.hand/from_timestamp": 0.0,
                "videos/observation.image.hand/to_timestamp": 20.0 / 30.0,
            },
            {
                "episode_index": 2,
                "length": 8,
                "task_success": True,
                "short_horizon_task": "task_b",
                "videos/observation.image.head/chunk_index": 0,
                "videos/observation.image.head/file_index": 2,
                "videos/observation.image.head/from_timestamp": 0.0,
                "videos/observation.image.head/to_timestamp": 8.0 / 30.0,
                "videos/observation.image.hand/chunk_index": 0,
                "videos/observation.image.hand/file_index": 2,
                "videos/observation.image.hand/from_timestamp": 0.0,
                "videos/observation.image.hand/to_timestamp": 8.0 / 30.0,
            },
        ]

        with (root / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_store_and_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_mock_dataset(root)

            store = AiroaMomaEpisodeStore.load(root, split_seed=123, val_ratio=0.34)
            self.assertEqual(len(store.episodes), 3)
            self.assertEqual(store.l_max_by_task["task_a"], 20)
            self.assertEqual(store.l_max_by_task["task_b"], 8)

            with patch("src.data.airoa_moma_vf_dataset.get_worker_video_cache", return_value=DummyVideoCache()):
                train_ds = AiroaMomaValueFrameDataset(
                    episode_store=store,
                    split="train",
                    frames_per_episode=3,
                    sample_seed=7,
                    num_bins=201,
                )
                val_ds = AiroaMomaValueFrameDataset(
                    episode_store=store,
                    split="val",
                    frames_per_episode=3,
                    sample_seed=7,
                    num_bins=201,
                )

                # 3 episodes total, val_ratio=0.34 -> val=1 episode, train=2 episodes.
                self.assertEqual(len(train_ds), 6)
                self.assertEqual(len(val_ds), 3)

                sample = train_ds[0]
                self.assertIn("head_image", sample)
                self.assertIn("hand_image", sample)
                self.assertIn("label_bin", sample)
                self.assertIn("v_norm", sample)
                self.assertTrue(-1.0 <= sample["v_norm"] <= 0.0)
                self.assertTrue(0 <= sample["label_bin"] <= 200)

    def test_compute_lmax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_mock_dataset(root)
            store = AiroaMomaEpisodeStore.load(root, split_seed=0, val_ratio=0.5)
            lmax = compute_l_max_by_task(store.episodes)
            self.assertEqual(lmax, {"task_a": 20, "task_b": 8})


if __name__ == "__main__":
    unittest.main()
