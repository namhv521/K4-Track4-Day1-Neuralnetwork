# Hướng dẫn làm bài Lab Day 1 — Neural Network

Tài liệu này dựa trên `README.md`, `GUIDE.md`, `RUBRIC.md`, code khung, notebook và hai script chia/chấm dữ liệu của repo. Đây là lộ trình thực hiện; các thông số thí nghiệm bên dưới là đề xuất cần kiểm chứng, không phải kết quả đã chạy.

## 1. Bài yêu cầu gì? Hiện đã làm đến đâu?

Bạn tự xây dựng MLP bằng PyTorch để phân loại **7 loại rừng** từ **54 đặc trưng** của Forest CoverType, rồi thực nghiệm để giải thích các yếu tố ảnh hưởng đến huấn luyện.

Chỉ số chính là **macro-F1**, vì dữ liệu mất cân bằng. Accuracy là chỉ số phụ. Bài không chỉ chấm mô hình dự đoán tốt: **85/100 điểm** thuộc quy trình, code, thí nghiệm, bảng, ảnh và lập luận; đánh giá eval chiếm 15 điểm.

Trạng thái đã kiểm tra trong checkout:

- Có dữ liệu gốc `data/covtype.csv.gz` và metadata `data/split_metadata.csv`.
- Chưa có thư mục `data/processed/`.
- Có `submission_2A202602853/code/`, nhưng các file giống hoàn toàn khung `code/`.
- Các hàm còn `NotImplementedError`; notebook còn TODO và chưa có output.
- Chưa có báo cáo, bảng kết quả, ảnh hay dự đoán eval trong thư mục bài nộp.

**Nơi làm việc:** hoàn thiện `submission_2A202602853/code/`. Giữ `code/` gốc làm mẫu tham khảo. Không cần copy lại và ghi đè thư mục bài nộp hiện có.

## 2. Đọc và sửa file nào?

| File | Vai trò | Việc cần làm |
|---|---|---|
| `README.md` | Luật và cấu trúc nộp | Đọc mục 3, 5, 6 |
| `RUBRIC.md` | Thang điểm | Đọc trước khi chọn thí nghiệm |
| `GUIDE.md` | Chi tiết Part 0–4 | Tra cứu khi triển khai |
| `scripts/split_data.py` | Sinh train/eval theo metadata | Chạy, không cần viết lại |
| `scripts/evaluate.py` | Chấm dự đoán cuối | Chạy sau khi chốt cấu hình |
| `submission_2A202602853/code/data.py` | Dữ liệu | Nạp, tách val, chuẩn hoá, chia batch |
| `submission_2A202602853/code/model.py` | MLP | Kiến trúc, khởi tạo, đếm tham số |
| `submission_2A202602853/code/optimizer.py` | Optimizer | Dùng `torch.optim`, đo/cắt gradient |
| `submission_2A202602853/code/train.py` | Pipeline | Loss, metric, vòng train, dự đoán |
| `submission_2A202602853/code/plots.py` | Biểu đồ | Ảnh từng lần chạy và so sánh nhóm |
| `submission_2A202602853/code/results_table.py` | Kết quả | JSON và bảng theo template |
| `submission_2A202602853/code/lab.ipynb` | Điều phối bài | Chạy Part 0–4, dự đoán/nhận xét |

Thứ tự triển khai: **data → model → optimizer/train → plots/lưu JSON → baseline → thí nghiệm → eval → bảng/báo cáo**.

## 3. Chuẩn bị môi trường và đường dẫn

Repo yêu cầu Python 3.10+, PyTorch 2.x, numpy, pandas, scikit-learn, matplotlib và openpyxl. Dùng môi trường riêng của bài hoặc môi trường notebook đã có đủ thư viện. Khi kiểm tra checkout, lệnh `python` đang trỏ tới virtualenv của một dự án khác; chưa kiểm tra các thư viện trong môi trường đó.

Từ **thư mục gốc repo**, sau khi môi trường sẵn sàng:

```powershell
python scripts/split_data.py
```

Script tạo:

```text
data/processed/train.npz
data/processed/eval.npz
```

Mở notebook trong `submission_2A202602853/code/`. Nếu working directory thực sự là thư mục này thì:

```python
REPO_ROOT = "../.."
OUT_DIR = ".."
```

Kiểm tra `os.getcwd()` trước khi chạy. Vị trí file notebook không tự bảo đảm working directory, nhất là khi dùng Colab. Cần bảo đảm:

- Import `data`, `model`, `train` lấy đúng module trong thư mục bài nộp.
- Dữ liệu đọc từ `<REPO_ROOT>/data/processed`.
- Kết quả ghi vào `<OUT_DIR>/results`, ảnh vào `<OUT_DIR>/figures`.
- Script `split_data.py` và `evaluate.py` chạy với `cwd=REPO_ROOT`, vì đường dẫn mặc định của chúng tương đối với working directory.

Trên Colab/Kaggle, làm theo ô thiết lập trong notebook và lưu kết quả sau mỗi lần chạy vào vị trí giữ được qua ngắt phiên. Chọn `cuda` khi có GPU, ngược lại `cpu`. In phiên bản PyTorch, device và tên GPU vào output.

## 4. Part 0 — Dữ liệu

Hoàn thiện các hàm trong `data.py` theo thứ tự:

1. `load_split`: đọc các khoá `X`, `y`, `row_id` từ NPZ.
2. `make_val_split`: tách 20% validation **từ train**, phân tầng theo nhãn, seed 42.
3. `fit_standardizer`: tính mean/std của **10 cột đầu trên phần train còn lại**; xử lý std bằng 0 để tránh chia cho 0.
4. `apply_standardizer`: áp dụng cùng mean/std cho train, val, eval; giữ nguyên 44 cột nhị phân.
5. `prepare_data`: gom các bước, tạo tensor đúng dtype/device và giữ `eval_row_id`.
6. `iterate_batches`: xáo train mỗi epoch, cắt batch, xử lý cả batch cuối nhỏ hơn bình thường.

Kích thước cần khớp:

| Tập | Số mẫu |
|---|---:|
| Train theo metadata | 464 809 |
| Train sau khi tách val | 371 847 |
| Validation | 92 962 |
| Eval | 116 203 |

`X` có 54 cột, dtype float32; `y` int64 và thuộc 0..6. Script chia đã trừ 1 từ nhãn gốc, không trừ thêm lần nữa.

Kiểm tra mean/std của 10 cột train sau chuẩn hoá gần 0/1 và in phân bố nhãn. Giữ split validation cố định khi đổi seed huấn luyện để đo riêng nhiễu từ khởi tạo/xáo batch.

**Qua bước này khi:** shape, dtype, nhãn đúng; chuẩn hoá không dùng thống kê val/eval; không sửa metadata.

## 5. Part 1 — Model và kiểm tra trước khi train lâu

Baseline bắt buộc:

```text
Linear(54,256) → ReLU → Linear(256,128) → ReLU → Linear(128,7)
```

Mọi Linear có bias; đầu ra là **logits thô**, không softmax trong model. Dropout khi dùng chỉ nằm sau ReLU lớp ẩn. Không thêm BatchNorm, LayerNorm hoặc residual.

Hoàn thiện `MLP`, `forward`, `init_weights`, `count_params`, `activation_stats`. Dùng `nn.Sequential`, `nn.init` sẵn có; không cần abstraction mới.

Chạy bốn kiểm tra trong notebook:

1. Đầu vào `(8,54)` cho đầu ra `(8,7)`.
2. `assert` số tham số M-base là **47 879**.
3. Đo CE bước 0 trên val và nhận xét so với `ln(7) ≈ 1,946`.
4. Tạo model riêng, tắt dropout, huấn luyện trên 20 mẫu đến accuracy 100% và loss gần 0; lưu đường cong. Kiểm tra gradient từng tham số sau backward.

**Lưu ý về loss bước 0:** `ln(7)` là CE khi xác suất các lớp đồng đều. Khởi tạo He ngẫu nhiên không bảo đảm logits đồng đều, nên loss đo được có thể lệch. Ghi số thật và kiểm tra nhãn, logits, chuẩn hoá, khởi tạo; không sửa số để khớp mốc.

Không dùng model đã quá khớp 20 mẫu làm baseline; khởi tạo lại model cho từng lần chạy.

## 6. Part 2 — Pipeline và baseline

Một hàm `run_experiment(cfg, data)` dùng chung cho mọi cấu hình. Hoàn thiện các hàm loss, evaluate, predict và optimizer trước, rồi mới ghép vòng train.

Baseline theo GUIDE:

| Thành phần | Giá trị |
|---|---|
| Hidden | `(256,128)` |
| Init | He, gọi `nn.init` rõ ràng |
| Loss | Cross-entropy |
| Optimizer | SGD + momentum 0.9 |
| Batch / epochs | 512 / 20 |
| Dropout / weight decay | 0 / 0 |
| Clip / precision | Không clip / FP32 |
| Learning rate | Chọn bằng validation |

Đề xuất thử lr `0.01`, `0.03`, `0.1` trước; đây là ứng viên, chưa biết giá trị nào tốt. Có thể chạy ngắn để sàng lọc nhưng phải ghi rõ budget. Các lần so sánh chính thức giữ cùng 20 epoch, cùng split và cùng seed, trừ yếu tố đang khảo sát.

Vòng train cần:

- Reset seed và tạo model/optimizer mới mỗi thí nghiệm.
- `model.train()` khi cập nhật; zero_grad → forward/loss → backward → đo/clip gradient → optimizer step.
- Dừng và ghi `diverged` nếu loss không hữu hạn.
- Cuối epoch, đo train loss và val loss ở `model.eval()` cùng `no_grad()`. Train có thể dùng tập con cố định 50 000 mẫu.
- Ghi val accuracy, macro-F1 đủ 7 lớp, gradient norm toàn cục trước clip, thời gian và peak GPU memory.
- Tính loss trung bình có trọng số theo số mẫu, tránh batch cuối làm sai trung bình.
- Chọn **best epoch bằng val loss thấp nhất**, sao chép state_dict độc lập; metric tóm tắt lấy ở epoch đó. Ghi riêng metric cuối nếu trường có tên `final_*`.
- Lưu JSON và PNG ngay sau mỗi lần chạy. JSON bỏ `best_state` vì tensor không phải dữ liệu JSON.

Sau khi chọn lr, chạy baseline với 2–3 seed, ví dụ 1, 2, 3. Tính mean/std của val macro-F1 và accuracy; dùng khoảng `2σ` làm mốc nhiễu theo mẫu báo cáo. Mốc này chỉ hỗ trợ nhận xét, không phải kiểm định thống kê chắc chắn.

**Qua bước này khi:** baseline chạy đủ, metric vượt mốc đoán đa số, có log/ảnh, và có thể giải thích đường train/val.

## 7. Part 3 — Chọn thí nghiệm có mục đích

Không bắt buộc một số lượng lần chạy cố định. Rubric có **7 chủ đề × 2 điểm**; mỗi chủ đề cần kết quả, dòng bảng, ảnh, dự đoán trước và đối chiếu sau để nhận đủ điểm độ phủ.

| Chủ đề | Đề xuất | Điều cần kết luận |
|---|---|---|
| Loss | CE so với MSE trên one-hot | So macro-F1/accuracy và hội tụ, không so giá trị loss trực tiếp |
| Optimizer | Baseline SGD momentum và Adam; mỗi loại ≥2 lr | So ở lr tốt nhất của từng loại |
| Hyper-parameter | Batch 128 so với 512 | Khác số bước/epoch và thời gian thế nào? |
| Dropout | Baseline 0 so với 0.3 | Khoảng cách train/val giảm không, F1 có tăng không? |
| Clipping | Cùng lr cao, không clip và có clip | Clip có kích hoạt và giảm bất ổn không? |
| Mixed precision | FP32 so với FP16 trên GPU | Thời gian, bộ nhớ, metric thực đo |
| Init | He so với Xavier và zeros | Loss bước 0, std kích hoạt, gradient thay đổi thế nào? |

Làm baseline ổn trước, rồi chạy lần lượt các chủ đề. Không cần grid search lớn. Nếu ít thời gian, ưu tiên optimizer/lr và vài thí nghiệm giải thích được; nêu rõ chủ đề chưa làm.

Mỗi lần chạy:

1. Ghi dự đoán 1–2 câu **trước khi chạy**.
2. Đặt `exp_id` duy nhất, mô tả yếu tố đổi.
3. Giữ các yếu tố khác như baseline. Riêng so optimizer cần chỉnh lr công bằng và ghi rõ quy trình.
4. Lưu `results/<exp_id>.json`, `figures/<exp_id>.png`, một dòng Experiments.
5. Viết nhận xét có số liệu, cơ chế và liên hệ nhiễu seed.

Ảnh riêng phải có ít nhất ba ô: train/val loss, val accuracy (nên thêm macro-F1), gradient norm trước clip. Thêm ảnh chồng so sánh theo nhóm.

### Các chi tiết dễ triển khai sai

- **MSE:** docstring `train.compute_loss` mô tả MSE giữa **logits và one-hot**. Theo cách này và ghi rõ trong báo cáo. Nếu chọn MSE trên softmax probabilities thì đó là định nghĩa khác, phải ghi rõ. Khi lấy mean trên N×7 phần tử, evaluate cần chia tổng squared error cho N×7 để cùng thang đo với train.
- **FP16:** unscale gradient trước khi đo norm, kể cả khi không clip; nếu không norm ghi log vẫn bị nhân bởi GradScaler. Chỉ bọc forward/loss bằng autocast.
- **BF16:** kiểm tra hỗ trợ GPU trước; nếu không chạy được ghi lý do thật, không điền kết quả giả.
- **Clipping:** chọn ngưỡng theo norm thực đo; muốn giải thích tần suất kích hoạt nên ghi thêm tỉ lệ bước norm vượt ngưỡng. So cặp lr cao có/không clip tại đúng cùng lr.
- **Zeros:** gradient bằng 0 ở lớp ẩn có thể là hiện tượng cần phân tích của thí nghiệm zeros. Không áp điều kiện mọi gradient khác 0 của model baseline cho trường hợp này.
- **Scheduler:** tuỳ chọn; chưa cần thêm nếu không khảo sát. Nếu giữ hàm khung không dùng, xử lý trường hợp `None` rõ ràng thay vì để `NotImplementedError`.
- **Kiến trúc:** nếu thử độ rộng/sâu, dùng M-wide hoặc M-deep được quy định. Ví dụ mạng lớn hơn trong mục tham chiếu của GUIDE không thay thế quy định kiến trúc.

## 8. Part 4 — Eval, bảng và báo cáo

### 8.1 Chốt cấu hình trước khi xem eval

Chọn cấu hình bằng validation; nạp checkpoint của epoch có val loss thấp nhất. Cố định seed đánh giá trước khi xem điểm eval. Nếu kết hợp nhiều kỹ thuật, ghi rõ và chạy lại trên val trước.

Dự đoán toàn bộ eval với `model.eval()` và `no_grad()`. CSV phải có đúng header `row_id,pred`, đủ 116 203 dòng dữ liệu, mỗi row_id một lần; nhãn nguyên 0..6. Row_id lấy từ NPZ, không dùng số thứ tự mới 0..116202.

Từ gốc repo:

```powershell
python scripts/evaluate.py --pred submission_2A202602853/predictions_eval.csv --out submission_2A202602853/eval_result.json
```

Làm tương tự cho baseline với file riêng để có số so sánh. Có thể giữ bằng chứng baseline trong `results/`. Chỉ lấy điểm eval cho baseline và cấu hình cuối; không chỉnh hyper-parameter sau khi thấy điểm eval.

Script JSON có accuracy, macro_f1, precision/recall/F1/support từng lớp và confusion matrix. Dùng chính các số này cho bảng/báo cáo. Phân tích lớp F1 thấp nhất, lớp thường bị nhầm và cơ sở của giải thích; phân biệt điều đo được với giả thuyết.

### 8.2 Bảng kết quả

Hoàn thiện `results_table.py` để điền từ JSON vào `templates/experiment_table_template.xlsx`:

- Giữ nguyên 4 sheet `Legend`, `Experiments`, `Seeds`, `Summary` và tên cột.
- Mỗi lần chạy đã thực hiện có một dòng, kể cả dò lr và lần diverged; ghi lý do thiếu metric vào notes.
- `figure_file` trỏ đúng `figures/<exp_id>.png`.
- Eval chỉ điền cho baseline và cấu hình cuối.
- Điền Seeds và nhận xét Summary; không chỉ ghi sheet Experiments.
- Khi thêm dòng vượt mẫu, kiểm tra công thức có được điền tiếp và tham chiếu đúng không; không ghi đè cột công thức.
- Mở bằng Excel/LibreOffice để tính lại và kiểm tra không có lỗi công thức.

### 8.3 Báo cáo

Viết `submission_2A202602853/REPORT.md` theo `templates/REPORT_TEMPLATE.md`, khoảng 4 trang:

1. Môi trường, dữ liệu, split, baseline.
2. Shape/tham số, loss bước 0, overfit 20 mẫu, gradient và nhiễu seed.
3. Chủ đề đã thử: dự đoán → số liệu/exp_id/ảnh → cơ chế → mức chênh so với nhiễu.
4. Baseline và cấu hình cuối trên eval, phân tích lỗi từng lớp.
5. Ba kiểm tra đầu tiên nếu loss không giảm sau 2 000 bước.
6. Hạn chế, kết quả bất ngờ, hướng kiểm chứng tiếp.

Mọi số phải tìm lại được trong bảng. Ảnh dùng đường dẫn tương đối như `![](figures/compare_optimizer.png)`. Xoá mục không làm và lời nhắc mẫu trước khi nộp.

## 9. Checklist hoàn thành

- [ ] Hoàn thiện các module trong thư mục bài nộp, không còn hàm khung chưa thực hiện.
- [ ] Split/chuẩn hoá đúng; metadata giữ nguyên.
- [ ] Model đúng shape/tham số, kiểm tra overfit 20 mẫu có đường cong.
- [ ] Baseline đúng cấu hình, đã chọn lr bằng val, có ít nhất 2 seed để đo nhiễu nếu muốn đủ điểm mục này.
- [ ] Mỗi lần chạy có cấu hình, kết quả, ảnh và nhận xét phù hợp.
- [ ] Chốt cấu hình bằng val trước khi đánh giá eval.
- [ ] CSV qua script chấm; JSON khớp bảng và báo cáo.
- [ ] Bảng giữ sheet/cột, công thức hoạt động, Summary/Seeds đã điền.
- [ ] Notebook Restart & Run All thành công và lưu output.
- [ ] Có REPORT.md, experiments.xlsx, predictions_eval.csv, eval_result.json, figures/, code/lab.ipynb cùng các module hoàn thiện.
- [ ] Đóng gói `submission_2A202602853/`; loại dữ liệu, checkpoint, cache và file tạm.

**Việc đầu tiên nên làm:** thiết lập đúng môi trường/working directory, chạy `split_data.py`, rồi hoàn thiện `data.py` và Part 0 của notebook. Chưa chạy nhiều thí nghiệm trước khi dữ liệu và kiểm tra model đều đạt.
