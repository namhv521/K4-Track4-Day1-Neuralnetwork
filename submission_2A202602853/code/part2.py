"""Validation-only LR screening (3 epochs), then 3 full baseline runs."""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from plots import plot_compare, plot_run
from results_table import save_result
from train import DEFAULT_CFG, run_experiment


def run_part2(data, out_dir):
    root = Path(out_dir)
    # Training never receives eval tensors, even though Part 0 prepares them.
    training = {k: data[k] for k in ('X_tr', 'y_tr', 'X_val', 'y_val')}
    # Bind resumable results to the actual data and pipeline, not filenames alone.
    digest = hashlib.sha256()
    for tensor in training.values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    for name in ('data.py', 'train.py', 'model.py', 'optimizer.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    fingerprint = digest.hexdigest()
    results_dir = root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = results_dir / 'part2_provenance.json'
    provenance = dict(fingerprint=fingerprint, torch=torch.__version__, device=str(data['X_tr'].device),
                      platform=platform.platform(), threads=torch.get_num_threads(),
                      train_rows=len(data['X_tr']), val_rows=len(data['X_val']),
                      train_loss_subset=min(50000, len(data['X_tr'])), split_seed=42)
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')

    def run(cfg):
        path = results_dir / f"{cfg['exp_id']}.json"
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        expected = json.loads(json.dumps(cfg))
        if (result is None or result.get('provenance') != provenance
                or result['cfg'] != expected or result['summary']['diverged']
                or len(result['history']['epoch']) != cfg['epochs']):
            result = run_experiment(cfg, training)
            result['provenance'] = provenance
            save_result(result, str(results_dir))
        else:
            print(f"Resume: {cfg['exp_id']} (verified data/code fingerprint)")
            # Submission rules exclude weight files. JSON resumes metrics, not weights.
            result['best_state'] = None
            print('Metrics-only resume: rerun run_experiment(cfg, data) before future final_eval.')
        plot_run(result, str(root / 'figures' / f"{cfg['exp_id']}.png"))
        return result

    trials = [run({**DEFAULT_CFG, 'lr': lr, 'epochs': 3,
                   'exp_id': f'lr-{lr:g}-s1', 'group': 'lr_screen',
                   'description': 'Validation-only LR screening; 3 epoch budget'})
              for lr in (0.01, 0.03, 0.1)]
    valid = [r for r in trials if not r['summary']['diverged']]
    if not valid:
        raise RuntimeError('All LR candidates diverged')
    chosen = min(valid, key=lambda r: r['summary']['best_val_loss'])['cfg']['lr']
    print('LR screening (selection = lowest validation loss, not eval):')
    for r in trials:
        print(r['cfg']['lr'], r['summary'])
    print('Selected LR:', chosen)
    baselines = [run({**DEFAULT_CFG, 'lr': chosen, 'seed': seed, 'exp_id': f'base-s{seed}'})
                 for seed in (1, 2, 3)]
    if any(r['summary']['diverged'] or len(r['history']['epoch']) != 20 for r in baselines):
        raise RuntimeError('Baseline incomplete or diverged; inspect saved JSON')
    majority = int(torch.bincount(training['y_tr'], minlength=7).argmax())
    majority_acc = float((training['y_val'] == majority).float().mean())
    summary = dict(selected_lr=chosen, lr_screen_epochs=3, baseline_epochs=20,
                   baseline_seeds=[1, 2, 3], majority_val_acc=majority_acc,
                   selection='minimum best_val_loss; same split/seed/budget',
                   std_definition='sample standard deviation, ddof=1', provenance=provenance)
    for metric in ('val_acc', 'val_macro_f1', 'best_val_loss'):
        values = [r['summary'][metric] for r in baselines]
        summary[metric] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                               noise_2sigma=float(2 * np.std(values, ddof=1)))
    summary['checks'] = dict(all_completed_20_epochs=True,
                              all_above_majority=all(r['summary']['val_acc'] > majority_acc for r in baselines),
                              all_final_above_majority=all(r['history']['val_acc'][-1] > majority_acc for r in baselines),
                              eval_used=False)
    (results_dir / 'part2_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    plot_compare(trials, 'val_loss', str(root / 'figures/compare_lr.png'), 'LR screening: same 3-epoch budget')
    plot_compare(baselines, 'val_loss', str(root / 'figures/compare_baseline_seeds.png'), 'Baseline: seed variation')
    print(json.dumps(summary, indent=2))
    return trials + baselines, summary


def baseline_notes(results, summary):
    lines = ['**Nhận xét baseline (kết quả thực đo Part 2):**\n',
             f"- Sàng lọc LR 0.01 / 0.03 / 0.1 cùng seed 1 và 3 epoch; chọn LR = {summary['selected_lr']} theo val loss thấp nhất. Budget ngắn chỉ dùng sàng lọc, không khẳng định LR tối ưu toàn cục.\n",
             '- Baseline mới: M-base, He, CE, SGD momentum 0.9, batch 512, FP32, dropout/weight decay = 0, không clip; 20 epoch × 3 seed. Split giữ seed 42; train loss đo eval trên 50,000 mẫu cố định.\n']
    for r in results:
        if r['cfg']['group'] != 'baseline':
            continue
        h, s = r['history'], r['summary']
        lines.append(f"- {r['cfg']['exp_id']}: step0 CE={s['step0_loss']:.6f}; best epoch={s['best_epoch']}, val loss={s['best_val_loss']:.6f}, accuracy={s['val_acc']:.2%}, macro-F1={s['val_macro_f1']:.6f}. Train loss {h['train_loss'][0]:.4f} → {h['train_loss'][-1]:.4f}; val loss {h['val_loss'][0]:.4f} → {h['val_loss'][-1]:.4f}; gap cuối={h['val_loss'][-1]-h['train_loss'][-1]:.4f}.\n")
        lines.append(f"  Val loss cuối {'cao hơn best: có dao động/giảm chất lượng sau best' if s['final_val_loss'] > s['best_val_loss'] else 'đạt best ở epoch cuối: còn cải thiện trong budget'}; cần đọc cả đường cong, chưa kết luận quá khớp chỉ từ một epoch.\n")
    for metric in ('val_acc', 'val_macro_f1', 'best_val_loss'):
        stat = summary[metric]
        lines.append(f"- {metric}: mean ± sample std = {stat['mean']:.6f} ± {stat['std']:.6f}; mốc nhiễu 2σ = {stat['noise_2sigma']:.6f}. Chỉ 3 seed, không phải kiểm định thống kê.\n")
    lines.extend([f"- Mốc đoán đa số val={summary['majority_val_acc']:.2%}; cả best và epoch cuối của 3 baseline đều vượt mốc: {summary['checks']['all_above_majority'] and summary['checks']['all_final_above_majority']}.\n",
                  '- Step0 không ép về ln(7)=1.945910: He có logits ngẫu nhiên khác 0, như chẩn đoán Part 1. Giữ đúng He của đề.\n',
                  '- Chạy CPU; peak GPU memory không áp dụng (null). Không dùng eval để chọn LR hoặc epoch. JSON và PNG lưu sau mỗi run.\n',
                  '\n![Baseline seed 1](../figures/base-s1.png)\n',
                  '\n![So sánh seed](../figures/compare_baseline_seeds.png)\n'])
    return lines
