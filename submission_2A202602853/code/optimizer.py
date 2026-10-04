"""optimizer.py — Optimizer và gradient clipping.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9,
                    betas=(0.9, 0.999), eps=1e-8):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    if lr is None or lr <= 0:
        raise ValueError("Choose a positive lr using validation")
    if name in ('sgd', 'sgd_momentum'):
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                               momentum=momentum if name == 'sgd_momentum' else 0)
    cls = torch.optim.Adam if name == 'adam' else torch.optim.AdamW
    return cls(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def clip_gradients(params, max_norm):
    """Return the global L2 norm before clipping (infinity disables clipping)."""
    if max_norm is not None and max_norm <= 0:
        raise ValueError('max_norm must be positive')
    return float(torch.nn.utils.clip_grad_norm_(params, float('inf') if max_norm is None else max_norm))
