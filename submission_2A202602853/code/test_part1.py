"""Run with: python -m unittest discover -s submission_2A202602853/code -p test_part1.py."""
import unittest
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from model import MLP, EXPECTED_PARAMS, activation_stats, count_params, init_weights


class Part1Tests(unittest.TestCase):
    def test_notebook_part1_checks_run_and_save_evidence(self):
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        from train import set_seed

        notebook = json.loads(Path(__file__).with_name('lab.ipynb').read_text(encoding='utf-8'))
        cell = next(c for c in notebook['cells'] if c['cell_type'] == 'code'
                    and ('# TODO 1: model' in ''.join(c['source'])
                         or '# ===== Part 1:' in ''.join(c['source'])))
        set_seed(42)
        X, y = torch.randn(20, 54), torch.arange(20) % 7
        with tempfile.TemporaryDirectory() as folder:
            namespace = dict(torch=torch, np=np, json=json, Path=Path, device='cpu',
                             OUT_DIR=folder, set_seed=set_seed, MLP=MLP,
                             EXPECTED_PARAMS=EXPECTED_PARAMS, count_params=count_params,
                             activation_stats=activation_stats,
                             data=dict(X_tr=X, y_tr=y, X_val=X, y_val=y))
            exec(''.join(cell['source']), namespace)
            summary = namespace['part1_summary']
            self.assertEqual(summary['params'], 47879)
            self.assertEqual(summary['overfit_acc'], 1.0)
            self.assertLess(summary['overfit_loss'], 0.01)
            self.assertAlmostEqual(summary['uniform_ce'], summary['ln7'], places=6)
            self.assertTrue(all(v > 0 for v in summary['gradient_norms'].values()))
            self.assertTrue(Path(folder, summary['figure_file']).is_file())
            result = json.loads(Path(folder, 'results/part1_checks.json').read_text())
            self.assertEqual(len(result['history']['loss']), summary['steps'] + 1)

    def test_allowed_architectures_return_raw_logits_and_have_expected_size(self):
        torch.manual_seed(42)
        for hidden, expected in EXPECTED_PARAMS.items():
            with self.subTest(hidden=hidden):
                model = MLP(hidden=hidden, dropout=0.3)
                self.assertEqual(count_params(model), expected)
                logits = model(torch.randn(8, 54))
                self.assertEqual(logits.shape, (8, 7))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertIsInstance(model.net[-1], nn.Linear)
                self.assertFalse(any(isinstance(m, nn.Softmax) for m in model.modules()))
                for i, layer in enumerate(model.net):
                    if isinstance(layer, nn.Dropout):
                        self.assertIsInstance(model.net[i - 1], nn.ReLU)

    def test_initializers_and_zero_logits_reference(self):
        for name in ('he', 'xavier', 'normal', 'zeros'):
            model = MLP(init=name)
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))
                    self.assertTrue(torch.isfinite(layer.weight).all())
            if name == 'zeros':
                logits = model(torch.randn(20, 54))
                self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
                loss = nn.functional.cross_entropy(logits, torch.arange(20) % 7)
                self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(7.)).item(), places=6)
            else:
                self.assertGreater(model.net[0].weight.std().item(), 0)
        model = MLP(init='default')
        before = {key: value.clone() for key, value in model.state_dict().items()}
        init_weights(model, 'default')
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        with self.assertRaises(ValueError):
            MLP(init='unknown')

    def test_gradients_flow_and_activation_stats_restore_training_mode(self):
        torch.manual_seed(42)
        model = MLP(dropout=0.3)
        X = torch.randn(20, 54)
        y = torch.arange(20) % 7
        loss = nn.functional.cross_entropy(model(X), y)
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.norm().item(), 0)
        stats = activation_stats(model, X)
        self.assertEqual(len(stats), 3)
        self.assertTrue(model.training)
        self.assertTrue(all(v > 0 for v in stats))
        self.assertEqual(stats, activation_stats(model, X))


if __name__ == '__main__':
    unittest.main()
