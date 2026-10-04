"""Part 2 checks: metrics, training isolation, divergence and artifacts."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')

from optimizer import build_optimizer, clip_gradients
from plots import plot_run
from results_table import save_result
from train import DEFAULT_CFG, compute_loss, evaluate, macro_f1_from_confusion, run_experiment


class Part2Tests(unittest.TestCase):
    def test_metrics_include_absent_classes_and_partial_batch(self):
        X = torch.zeros(5, 7)
        y = torch.tensor([0, 0, 0, 1, 1])
        scores = evaluate(nn.Identity(), X, y, batch_size=3)
        self.assertAlmostEqual(scores['loss'], np.log(7), places=6)
        self.assertAlmostEqual(scores['acc'], 0.6)
        self.assertAlmostEqual(scores['macro_f1'], 0.75 / 7)
        self.assertAlmostEqual(compute_loss(X, y, 'mse').item(), 1 / 7)

    def test_gradient_norm_is_measured_before_clip(self):
        p = nn.Parameter(torch.ones(2))
        p.grad = torch.tensor([3., 4.])
        self.assertAlmostEqual(clip_gradients([p], 1), 5)
        self.assertAlmostEqual(p.grad.norm().item(), 1, places=5)
        for name in ('sgd', 'sgd_momentum', 'adam', 'adamw'):
            self.assertIsInstance(build_optimizer(name, [p], 0.01), torch.optim.Optimizer)

    def test_training_uses_only_validation_and_saves_artifacts(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        X = torch.randn(28, 54)
        y = torch.arange(28) % 7
        data = dict(X_tr=X, y_tr=y, X_val=X[:13], y_val=y[:13])
        cfg = {**DEFAULT_CFG, 'lr': 0.03, 'epochs': 3, 'batch': 11}
        result = run_experiment(cfg, data)
        self.assertEqual(result['history']['epoch'], [1, 2, 3])
        best = int(np.argmin(result['history']['val_loss']))
        self.assertEqual(result['summary']['best_epoch'], best + 1)
        self.assertEqual(result['summary']['val_macro_f1'], result['history']['val_macro_f1'][best])
        again = run_experiment(cfg, {**data, 'X_eval': None, 'y_eval': None})
        self.assertEqual(result['history']['val_loss'], again['history']['val_loss'])
        with tempfile.TemporaryDirectory() as folder:
            path = save_result(result, folder)
            self.assertNotIn('best_state', json.loads(Path(path).read_text()))
            plot_run(result, str(Path(folder) / 'run.png'))
            self.assertGreater((Path(folder) / 'run.png').stat().st_size, 1000)

    def test_saved_run_keeps_its_own_provenance(self):
        # A changed run may stop halfway: global metadata cannot authenticate old JSON.
        result = dict(cfg={'exp_id': 'run'}, history={}, summary={},
                      provenance={'fingerprint': 'old-data-code'})
        with tempfile.TemporaryDirectory() as folder:
            path = save_result(result, folder)
            loaded = json.loads(Path(path).read_text())
            self.assertEqual(loaded.get('provenance'), result['provenance'])

    def test_nonfinite_loss_stops_cleanly(self):
        X = torch.full((7, 54), float('nan'))
        y = torch.arange(7)
        result = run_experiment({**DEFAULT_CFG, 'lr': 0.01, 'epochs': 3},
                                dict(X_tr=X, y_tr=y, X_val=X, y_val=y))
        self.assertTrue(result['summary']['diverged'])
        self.assertIsNone(result['best_state'])


if __name__ == '__main__':
    unittest.main()
