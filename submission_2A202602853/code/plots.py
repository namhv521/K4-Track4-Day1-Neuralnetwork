"""plots.py — Biểu đồ từng run và so sánh các run.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result, path):
    from pathlib import Path
    cfg, h = result['cfg'], result['history']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for key in ('train_loss', 'val_loss'):
        axes[0].plot(h['epoch'], h[key], label=key)
    for key in ('val_acc', 'val_macro_f1'):
        axes[1].plot(h['epoch'], h[key], label=key)
    axes[2].plot(h['epoch'], h['grad_norm'], label='L2 norm before clipping')
    for ax, label in zip(axes, ('Loss', 'Validation score', 'Gradient norm')):
        ax.set(xlabel='Epoch', ylabel=label)
        ax.axvline(result['summary']['best_epoch'], color='gray', linestyle='--', label='Best epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} batch={cfg['batch']} "
                 f"hidden={cfg['hidden']} init={cfg['init']} {cfg['precision']} seed={cfg['seed']}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_compare(results, metric, path, title=''):
    from pathlib import Path
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(result['history']['epoch'], result['history'][metric], label=result['cfg']['exp_id'])
    ax.set(xlabel='Epoch', ylabel=metric, title=title or metric)
    ax.legend()
    ax.grid(alpha=0.2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
