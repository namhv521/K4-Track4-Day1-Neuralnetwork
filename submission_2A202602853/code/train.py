"""train.py — Pipeline huấn luyện và xuất dự đoán Part 2–4.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def macro_f1_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.float64)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * np.diag(cm), denom, out=np.zeros(7), where=denom > 0).mean())


@torch.no_grad()
def predict(model, X, batch_size=8192):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    model.eval()
    if len(X) == 0:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat([model(X[i:i+batch_size]).argmax(1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name='ce', batch_size=8192):
    if len(X) == 0 or len(X) != len(y) or batch_size <= 0:
        raise ValueError('Evaluation needs nonempty aligned X/y and positive batch size')
    model.eval()
    total = torch.zeros((), device=X.device)
    cm = torch.zeros(49, dtype=torch.int64, device=X.device)
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start+batch_size], y[start:start+batch_size]
        logits = model(xb)
        total += compute_loss(logits, yb, loss_name) * len(yb)
        cm += torch.bincount(7 * yb + logits.argmax(1), minlength=49)
    matrix = cm.reshape(7, 7).cpu().numpy()
    return dict(loss=float(total / len(X)), acc=float(np.trace(matrix) / len(X)),
                macro_f1=macro_f1_from_confusion(matrix))


def compute_loss(logits, y, loss_name):
    if loss_name == 'ce':
        return F.cross_entropy(logits, y)
    if loss_name == 'mse':
        # Mean across both samples and the seven output classes.
        return F.mse_loss(logits, F.one_hot(y, 7).to(logits.dtype))
    raise ValueError(f'Unknown loss: {loss_name}')


def run_experiment(cfg, data):
    """Train using train/val only; retain an independent best-validation checkpoint."""
    cfg = {**DEFAULT_CFG, **cfg}
    cfg['hidden'] = tuple(cfg['hidden'])
    if cfg['epochs'] <= 0 or cfg['batch'] <= 0:
        raise ValueError('epochs and batch must be positive')
    precision = cfg['precision']
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError(f'Unknown precision: {precision}')
    X, y = data['X_tr'], data['y_tr']
    device = X.device
    if precision == 'fp16' and device.type != 'cuda':
        raise ValueError('FP16 training requires CUDA')
    set_seed(cfg['seed'])
    model = MLP(cfg['hidden'], cfg['dropout'], cfg['init']).to(device)
    assert count_params(model) == EXPECTED_PARAMS[cfg['hidden']]
    optimizer = build_optimizer(cfg['optimizer'], model.parameters(), cfg['lr'],
                                cfg['weight_decay'], cfg['momentum'])
    scaler = torch.amp.GradScaler(device.type, enabled=precision == 'fp16')
    generator = torch.Generator(device=device).manual_seed(cfg['seed'])
    # A fixed 50k training subset keeps epoch evaluation affordable and comparable.
    subset = torch.randperm(len(X), generator=torch.Generator(device=device).manual_seed(42),
                            device=device)[:50000]
    history = {k: [] for k in ('epoch', 'train_loss', 'val_loss', 'val_acc',
                               'val_macro_f1', 'grad_norm', 'epoch_time_s')}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    initial = evaluate(model, data['X_val'], data['y_val'], cfg['loss'])
    initial_activation_std = activation_stats(model, data['X_val'][:512])
    clipped_steps, total_steps = 0, 0
    step0 = initial['loss']
    diverged = not np.isfinite(step0)
    best_loss, best_epoch, best_state, best_scores = float('inf'), 0, None, None
    for epoch in range(1, cfg['epochs'] + 1):
        if diverged:
            break
        sync()
        started = time.perf_counter()
        model.train()
        norms = []
        for xb, yb in iterate_batches(X, y, cfg['batch'], generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if precision == 'fp16' else torch.bfloat16,
                                enabled=precision != 'fp32'):
                loss = compute_loss(model(xb), yb, cfg['loss'])
            if not torch.isfinite(loss):
                diverged = True
                break
            scaler.scale(loss).backward()
            # Unscale even without clipping so the logged norm is in true units.
            scaler.unscale_(optimizer)
            norm = clip_gradients(model.parameters(), cfg['clip_norm'])
            if not np.isfinite(norm):
                diverged = True
                break
            norms.append(norm)
            total_steps += 1
            clipped_steps += int(cfg['clip_norm'] is not None and norm > cfg['clip_norm'])
            scaler.step(optimizer)
            scaler.update()
        if diverged:
            break
        train_scores = evaluate(model, X[subset], y[subset], cfg['loss'])
        val = evaluate(model, data['X_val'], data['y_val'], cfg['loss'])
        sync()
        elapsed = time.perf_counter() - started
        if not np.isfinite(train_scores['loss']) or not np.isfinite(val['loss']):
            diverged = True
            break
        values = (epoch, train_scores['loss'], val['loss'], val['acc'], val['macro_f1'],
                  float(np.mean(norms)), elapsed)
        for key, value in zip(history, values):
            history[key].append(value)
        if val['loss'] < best_loss:
            best_loss, best_epoch, best_scores = val['loss'], epoch, val.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{cfg['exp_id']} epoch {epoch}/{cfg['epochs']}: train={train_scores['loss']:.5f} "
              f"val={val['loss']:.5f} acc={val['acc']:.5f} F1={val['macro_f1']:.5f} ({elapsed:.1f}s)", flush=True)
    summary = dict(step0_loss=step0 if np.isfinite(step0) else None,
                   best_val_loss=best_loss if best_state is not None else None,
                   best_epoch=best_epoch,
                   final_train_loss=history['train_loss'][-1] if history['epoch'] else None,
                   final_val_loss=history['val_loss'][-1] if history['epoch'] else None,
                   val_acc=best_scores['acc'] if best_scores else None,
                   val_macro_f1=best_scores['macro_f1'] if best_scores else None,
                   time_per_epoch_s=float(np.mean(history['epoch_time_s'])) if history['epoch'] else None,
                   peak_mem_MB=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None,
                   diverged=diverged, initial_activation_std=initial_activation_std,
                   clipped_steps=clipped_steps, total_steps=total_steps,
                   clip_fraction=clipped_steps / total_steps if total_steps else 0.0)
    return dict(cfg=cfg, history=history, summary=summary, best_state=best_state)


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    ids, labels = np.asarray(row_id), np.asarray(preds)
    if (ids.ndim != 1 or labels.shape != ids.shape or len(ids) == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(labels.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or (ids < 0).any()
            or ((labels < 0) | (labels > 6)).any()):
        raise ValueError('Predictions require aligned unique integer row IDs and integer labels 0..6')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, np.column_stack((ids, labels)), fmt='%d', delimiter=',',
               header='row_id,pred', comments='')


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    if result.get('best_state') is None:
        raise ValueError('Metrics-only result: retrain to obtain best_state before final evaluation')
    if {**cfg, 'hidden': tuple(cfg['hidden'])} != {**result['cfg'], 'hidden': tuple(result['cfg']['hidden'])}:
        raise ValueError('Evaluation config must match the trained result')
    model = MLP(cfg['hidden'], cfg['dropout'], cfg['init']).to(data['X_eval'].device)
    model.load_state_dict(result['best_state'])
    preds = predict(model, data['X_eval']).cpu().numpy()
    write_predictions(data['eval_row_id'], preds, pred_path)
