"""Submission contracts: row IDs, validation selection, and spreadsheet formulas."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openpyxl
import torch

from results_table import load_results, save_result, to_row, write_xlsx
from train import DEFAULT_CFG, final_eval, run_experiment, write_predictions
from part3 import select_candidate
from part4 import choose_output_dir


class SubmissionTests(unittest.TestCase):
    def test_changed_runtime_preserves_sealed_outputs(self):
        x = torch.zeros(7, 54)
        y = torch.arange(7)
        data = dict(X_tr=x, y_tr=y, X_val=x, y_val=y)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'submission'
            (root / 'results').mkdir(parents=True)
            lock = root / 'results/final_selection.json'
            lock.write_text('{"provenance": {"device": "cuda"}}')
            original = lock.read_bytes()
            output = choose_output_dir(data, str(root))
            self.assertNotEqual(output, str(root))
            self.assertTrue(Path(output, 'results').is_dir())
            self.assertEqual(lock.read_bytes(), original)

    def test_selection_uses_f1_not_cross_loss_scale_or_diverged_runs(self):
        def result(name, f1, loss, diverged=False):
            return dict(cfg=dict(exp_id=name, epochs=20), history=dict(epoch=list(range(20))),
                        summary=dict(val_macro_f1=f1, val_acc=0.9, best_val_loss=loss, diverged=diverged))
        ce = result('ce', 0.85, 0.2)
        mse = result('mse', 0.8, 0.001)
        broken = result('broken', 0.99, 0.001, True)
        self.assertIs(select_candidate([ce, mse, broken]), ce)
        with self.assertRaises(ValueError):
            select_candidate([broken])

    def test_predictions_reject_bad_ids_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pred.csv'
            for ids, labels in (([2, 2], [0, 1]), ([2, 3], [0, 7]),
                                ([2, 3], [0.5, 1]), ([2], [0, 1])):
                with self.assertRaises(ValueError):
                    write_predictions(np.array(ids), np.array(labels), str(path))
            write_predictions(np.array([9, 2]), np.array([6, 0]), str(path))
            with path.open() as stream:
                self.assertEqual(list(csv.reader(stream)), [['row_id', 'pred'], ['9', '6'], ['2', '0']])

    def test_final_eval_requires_weights_and_matching_config(self):
        torch.set_num_threads(2)
        x = torch.randn(28, 54)
        y = torch.arange(28) % 7
        data = dict(X_tr=x, y_tr=y, X_val=x, y_val=y, X_eval=x,
                    eval_row_id=np.arange(28))
        result = run_experiment(dict(lr=0.01, epochs=1), data)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'pred.csv')
            final_eval(result['cfg'], result, data, path)
            self.assertEqual(len(Path(path).read_text().splitlines()), 29)
            with self.assertRaises(ValueError):
                final_eval({**result['cfg'], 'dropout': 0.5}, result, data, path)
            result['best_state'] = None
            with self.assertRaises(ValueError):
                final_eval(result['cfg'], result, data, path)

    def test_result_loading_filters_metadata_and_preserves_formulas(self):
        result = dict(cfg={**DEFAULT_CFG, 'lr': 0.1}, history={},
                      summary=dict(val_macro_f1=0.8, val_acc=0.9, diverged=False))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_result(result, str(root))
            (root / 'part2_summary.json').write_text('{"selected_lr": 0.1}')
            (root / 'part1_checks.json').write_text('{"cfg": {}, "history": {}, "summary": {}}')
            loaded = load_results(str(root))
            self.assertEqual(len(loaded), 1)
            row = to_row(loaded[0], dict(accuracy=0.91, macro_f1=0.81))
            self.assertEqual(row['eval_macro_f1'], 0.81)
            template = Path(__file__).resolve().parents[2] / 'templates/experiment_table_template.xlsx'
            output = root / 'table.xlsx'
            write_xlsx([row], str(template), str(output))
            wb = openpyxl.load_workbook(output)
            self.assertEqual(wb['Experiments']['A2'].value, 'base-s1')
            self.assertTrue(wb['Experiments']['AF2'].value.startswith('='))
            self.assertIsNone(wb['Experiments']['A3'].value)
            self.assertEqual(wb['Seeds']['A2'].value, 'base-s1')


if __name__ == '__main__':
    unittest.main()
