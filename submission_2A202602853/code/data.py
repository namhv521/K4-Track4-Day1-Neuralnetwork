"""data.py — Chuẩn bị dữ liệu cho Part 0.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    root = Path(processed_dir)
    with np.load(root / "train.npz") as train, np.load(root / "eval.npz") as ev:
        X_train, y_train = train["X"], train["y"]
        X_eval, y_eval, row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in ((X_train, y_train), (X_eval, y_eval)):
        assert X.ndim == 2 and X.shape[1] == 54 and X.dtype == np.float32
        assert y.shape == (len(X),) and y.dtype == np.int64
        assert np.isfinite(X).all() and ((y >= 0) & (y < 7)).all()
    assert row_id.shape == y_eval.shape and row_id.dtype == np.int64
    assert len(np.unique(row_id)) == len(row_id)
    return X_train, y_train, X_eval, y_eval, row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    # Float64 accumulation avoids rounding error on hundreds of thousands of rows.
    numeric = X_tr[:, :N_NUMERIC]
    mean = numeric.mean(axis=0, dtype=np.float64)
    std = numeric.std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    scaled = X.astype(np.float32, copy=True)
    safe_std = np.where(std == 0, 1.0, std)
    scaled[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / safe_std
    return scaled


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_full, y_full, X_eval, y_eval, row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    result = {"eval_row_id": row_id, "mean": mean, "std": std}
    for name, X, y in (("tr", X_tr, y_tr), ("val", X_val, y_val),
                       ("eval", X_eval, y_eval)):
        result[f"X_{name}"] = torch.as_tensor(
            apply_standardizer(X, mean, std), dtype=torch.float32, device=device
        )
        result[f"y_{name}"] = torch.as_tensor(y, dtype=torch.int64, device=device)
        print(f"{name}: X={tuple(X.shape)}, y={tuple(y.shape)}, device={device}")
    majority = np.bincount(y_tr, minlength=7).argmax()
    print(f"Majority class = {majority}; val accuracy = {(y_val == majority).mean():.6f}")
    return result


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if shuffle:
        indices = torch.randperm(len(X), generator=generator, device=X.device)
    else:
        indices = torch.arange(len(X), device=X.device)
    # Include the final partial batch: no samples are discarded.
    for start in range(0, len(X), batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y[idx]
