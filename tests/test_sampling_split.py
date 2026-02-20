import random
import unittest

from src.data.airoa_moma_vf_dataset import sample_episode_steps, split_episode_indices


class SamplingSplitTest(unittest.TestCase):
    def test_sample_episode_steps_without_replacement(self):
        rng = random.Random(123)
        steps = sample_episode_steps(length=10, k=5, rng=rng)
        self.assertEqual(len(steps), 5)
        self.assertEqual(len(set(steps)), 5)

    def test_sample_episode_steps_with_replacement(self):
        rng = random.Random(123)
        steps = sample_episode_steps(length=3, k=8, rng=rng)
        self.assertEqual(len(steps), 8)
        self.assertTrue(all(0 <= x < 3 for x in steps))

    def test_split_episode_indices_ratio_and_determinism(self):
        episodes = list(range(100))
        train_a, val_a = split_episode_indices(episodes, seed=42, val_ratio=0.05)
        train_b, val_b = split_episode_indices(episodes, seed=42, val_ratio=0.05)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(val_a), 5)
        self.assertEqual(len(train_a), 95)
        self.assertEqual(set(train_a).intersection(val_a), set())
        self.assertEqual(set(train_a).union(val_a), set(episodes))


if __name__ == "__main__":
    unittest.main()
