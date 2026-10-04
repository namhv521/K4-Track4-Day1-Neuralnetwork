"""Controlled experiments on one fixed training/validation split; no eval access."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from plots import plot_compare, plot_run
from results_table import save_result
from train import run_experiment


def select_candidate(results):
    """Compare CE/MSE using F1, never their differently scaled loss values."""
    valid = [r for r in results if not r['summary']['diverged']
             and r['summary'].get('val_macro_f1') is not None
             and len(r['history']['epoch']) == r['cfg']['epochs']]
    if not valid:
        raise ValueError('No completed validation candidate')
    return max(valid, key=lambda r: (r['summary']['val_macro_f1'], r['summary']['val_acc']))


def run_part3(data, out_dir, part2_results, part2_summary):
    root = Path(out_dir)
    training = {k: data[k] for k in ('X_tr', 'y_tr', 'X_val', 'y_val')}
    base = next(r for r in part2_results if r['cfg']['exp_id'] == 'base-s1')
    cfg = base['cfg']
    noise = part2_summary['val_macro_f1']['noise_2sigma']
    amp_precision = 'bf16' if data['X_tr'].device.type == 'cpu' or torch.cuda.is_bf16_supported() else 'fp16'
    # Half the baseline mean pre-clip norm makes activation measurable, not assumed.
    clip = round(float(np.mean(base['history']['grad_norm'])) / 2, 3)
    menu = [
        ('opt-sgdm-0.03', 'optimizer', {'lr': 0.03},
         'SGD momentum LR 0.03 may converge more slowly than 0.1.',
         'Momentum accumulates gradients; LR controls the size of each update.'),
        ('opt-adam-0.001', 'optimizer', {'optimizer': 'adam', 'lr': 0.001},
         'Adam may converge faster through per-parameter adaptive updates.',
         'Adam uses bias-corrected first/second moments; optimizer and its LR change together for fair tuning.'),
        ('opt-adam-0.003', 'optimizer', {'optimizer': 'adam', 'lr': 0.003},
         'Larger Adam LR may accelerate early fitting but increase oscillation.',
         'Compare each optimizer at its own best validation LR with the same 20-epoch budget.'),
        ('loss-mse', 'loss', {'loss': 'mse'},
         'MSE on raw logits versus one-hot targets may fit classification less effectively than CE.',
         'MSE averages (logit-one_hot)^2 over B x 7 without 1/2; it is not MSE on softmax probabilities. CE gradient is p-y; logit MSE gradient is 2*(z-y)/7. Compare F1, not loss scales.'),
        ('batch-2048', 'hparam', {'batch': 2048},
         'Larger batch reduces updates per epoch and may slow convergence at fixed LR.',
         'Batch 2048 changes update count, not just memory: ceil(N/2048) vs ceil(N/512) per epoch.'),
        ('drop-0.1', 'dropout', {'dropout': 0.1},
         'Dropout may reduce the train-val gap but hurt a baseline still improving at epoch 20.',
         'Dropout adds training noise and reduces co-adaptation; train loss is measured in eval mode.'),
        ('clip-normal', 'clipping', {'clip_norm': clip},
         'Clipping should activate at normal LR and may slow learning.',
         'Global norm threshold is half the baseline epoch-mean norm; actual clipped-step fraction is recorded.'),
        ('clip-high-none', 'clipping', {'lr': cfg['lr'] * 10},
         'Ten-fold LR may cause large gradients, unstable loss, or divergence.',
         'Stress-test reference changes LR; compare clipping only against this same high-LR run.'),
        ('clip-high-on', 'clipping', {'lr': cfg['lr'] * 10, 'clip_norm': clip},
         'At the same high LR clipping may limit spikes, but cannot guarantee stable convergence.',
         'Compared with clip-high-none only clip_norm changes; norms are logged before clipping.'),
        (f'amp-{amp_precision}', 'amp', {'precision': amp_precision},
         'Mixed precision may not improve end-to-end speed on this small MLP; numerical trajectories may change.',
         'BF16 has an FP32-like exponent range; FP16 has narrower range and needs loss scaling. Parameters and evaluation remain FP32.'),
        ('init-zeros', 'init', {'init': 'zeros'},
         'Zero initialization leaves hidden ReLU gradients zero; only output bias can learn class priors.',
         'Identical zero neurons preserve symmetry and ReLU derivative at zero is zero.'),
        ('init-normal', 'init', {'init': 'normal'},
         'Small normal weights shrink activations and may delay convergence.',
         'Normal std=0.01 is independent of fan-in; repeated layers can attenuate signal.'),
        ('init-xavier', 'init', {'init': 'xavier'},
         'Xavier may shrink ReLU activations relative to He; a shallow MLP may still train well.',
         'Xavier normal variance=2/(fan_in+fan_out); He variance=2/fan_in for ReLU.'),
    ]
    planned = []
    for exp_id, group, changes, hypothesis, mechanism in menu:
        planned.append(dict(cfg={**cfg, **changes, 'exp_id': exp_id, 'group': group,
                                 'description': hypothesis}, hypothesis=hypothesis, mechanism=mechanism))
    # Hypotheses are persisted before any Part 3 run starts.
    plan_path = root / 'results/part3_plan.json'
    plan_path.write_text(json.dumps(planned, indent=2) + '\n', encoding='utf-8')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {**part2_summary['provenance'], 'part3_sha256': digest}

    def run(config):
        path = root / 'results' / f"{config['exp_id']}.json"
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        if (result is None or result.get('provenance') != provenance
                or result['cfg'] != json.loads(json.dumps(config))):
            result = run_experiment(config, training)
            result['provenance'] = provenance
            save_result(result, str(root / 'results'))
        else:
            result['best_state'] = None
            print('Resume verified:', config['exp_id'])
        plot_run(result, str(root / 'figures' / f"{config['exp_id']}.png"))
        return result

    results = [run(item['cfg']) for item in planned]
    candidate = select_candidate([base] + results)
    selected = {**candidate['cfg'], 'epochs': 40, 'group': 'final',
                'description': 'Validation-selected configuration; extended 40-epoch budget'}
    # Three repeat runs measure noise for the final configuration, not a seed ensemble.
    finals = [run({**selected, 'exp_id': f'final-s{seed}', 'seed': seed}) for seed in (1, 2, 3)]
    final = select_candidate(finals)
    summary = dict(candidate_exp_id=candidate['cfg']['exp_id'],
                   final_exp_id=final['cfg']['exp_id'], selection='maximum validation macro-F1 at minimum-loss checkpoint; accuracy tie-break',
                   extended_budget=40, baseline_noise_2sigma=noise, eval_used=False,
                   clip_threshold=clip, provenance=provenance,
                   fp16_status='available' if torch.cuda.is_available() else 'unavailable: FP16 training requires CUDA; measured BF16 on CPU',
                   amp_exp_id=f'amp-{amp_precision}',
                   bf16_status='CPU autocast measured' if data['X_tr'].device.type == 'cpu' else ('CUDA supported' if torch.cuda.is_bf16_supported() else 'CUDA unsupported; measured FP16 instead'),
                   final_f1_mean=float(np.mean([r['summary']['val_macro_f1'] for r in finals])),
                   final_f1_std=float(np.std([r['summary']['val_macro_f1'] for r in finals], ddof=1)))
    (root / 'results/part3_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for group in ('loss', 'optimizer', 'hparam', 'dropout', 'clipping', 'amp', 'init', 'final'):
        comparison = [base] + [r for r in results + finals if r['cfg']['group'] == group]
        # F1 is comparable across losses; separate loss plots already exist per run.
        plot_compare(comparison, 'val_macro_f1', str(root / f'figures/compare_{group}.png'), f'{group}: validation macro-F1')
    print(json.dumps(summary, indent=2))
    return results + finals, summary
