"""Run with: python -m unittest discover -s submission_2A202602853/code -p test_part0.py."""
import tempfile
import random
import unittest
from pathlib import Path

import numpy as np
import torch

from data import (apply_standardizer, fit_standardizer, iterate_batches,
                  load_split, make_val_split, prepare_data)
from train import set_seed


class Part0Tests(unittest.TestCase):
    def test_seed_reproduces_python_numpy_and_torch_draws(self):
        draws = []
        for _ in range(2):
            set_seed(42)
            draws.append((random.random(), np.random.rand(), torch.rand(3)))
        self.assertEqual(draws[0][:2], draws[1][:2])
        self.assertTrue(torch.equal(draws[0][2], draws[1][2]))

    def test_standardization_uses_training_statistics_and_preserves_binary_columns(self):
        train = np.zeros((3, 54), dtype=np.float32)
        train[:, :9] = np.array([1, 3, 5])[:, None]
        train[:, 9] = 7  # A constant column must stay finite.
        train[:, 10:] = np.arange(44) % 2
        before = train.copy()
        mean, std = fit_standardizer(train)
        scaled = apply_standardizer(train, mean, std)
        np.testing.assert_allclose(scaled[:, :9].mean(0), 0, atol=1e-6)
        np.testing.assert_allclose(scaled[:, :9].std(0), 1, atol=1e-6)
        self.assertTrue(np.isfinite(scaled).all())
        np.testing.assert_array_equal(scaled[:, 10:], train[:, 10:])
        np.testing.assert_array_equal(train, before)
        held_out = train.copy()
        held_out[:, :9] += 100
        shifted = apply_standardizer(held_out, mean, std)
        self.assertGreater(float(shifted[:, :9].mean()), 10)

    def test_preparation_matches_train_only_statistics_and_keeps_eval_row_ids(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(70, 54)).astype(np.float32)
        X[:, 10:] = rng.integers(0, 2, size=(70, 44))
        y = np.tile(np.arange(7, dtype=np.int64), 10)
        E = X[:7].copy()
        E[:, :10] += 1000  # Eval must not influence fitted statistics.
        ids = np.arange(100, 107, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / 'train.npz', X=X, y=y)
            np.savez(Path(tmp) / 'eval.npz', X=E, y=y[:7], row_id=ids)
            loaded = load_split(tmp)
            np.testing.assert_array_equal(loaded[-1], ids)
            tr, yt, val, yv = make_val_split(X, y)
            tr2, yt2, val2, yv2 = make_val_split(X, y)
            np.testing.assert_array_equal(tr, tr2)
            np.testing.assert_array_equal(yv, yv2)
            self.assertEqual((len(tr), len(val)), (56, 14))
            np.testing.assert_array_equal(np.bincount(yt), np.full(7, 8))
            np.testing.assert_array_equal(np.bincount(yv), np.full(7, 2))
            data = prepare_data('cpu', processed_dir=tmp)
            mean, std = fit_standardizer(tr)
            np.testing.assert_allclose(data['X_eval'].numpy(), apply_standardizer(E, mean, std))
            np.testing.assert_array_equal(data['X_val'].numpy()[:, 10:], val[:, 10:])
            np.testing.assert_array_equal(data['eval_row_id'], ids)
            self.assertEqual(data['X_tr'].dtype, torch.float32)
            self.assertEqual(data['y_tr'].dtype, torch.int64)

    def test_batches_include_each_row_once_keep_pairs_and_reproduce_shuffle(self):
        X = torch.arange(11).reshape(-1, 1)
        y = X[:, 0] * 2
        batches = list(iterate_batches(X, y, 4, shuffle=False))
        self.assertEqual([len(x) for x, _ in batches], [4, 4, 3])
        self.assertTrue(torch.equal(torch.cat([x for x, _ in batches]), X))
        shuffled = []
        for _ in range(2):
            g = torch.Generator(device='cpu').manual_seed(7)
            pairs = list(iterate_batches(X, y, 4, generator=g))
            for xb, yb in pairs:
                self.assertTrue(torch.equal(xb[:, 0] * 2, yb))
            shuffled.append(torch.cat([xb[:, 0] for xb, _ in pairs]))
        self.assertTrue(torch.equal(shuffled[0], shuffled[1]))
        self.assertTrue(torch.equal(shuffled[0].sort().values, X[:, 0]))
        with self.assertRaises(ValueError):
            list(iterate_batches(X, y, 0))


if __name__ == '__main__':
    unittest.main()
