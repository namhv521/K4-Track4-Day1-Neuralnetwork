"""Freeze validation selection, score once, then build submission artifacts."""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from plots import plot_run
from results_table import load_results, save_result, to_row, write_xlsx
from train import final_eval, run_experiment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_output_dir(data, out_dir):
    """Preflight before Part 1 writes: preserve sealed results on a different runtime."""
    root = Path(out_dir)
    lock = root / 'results/final_selection.json'
    if not lock.exists():
        return str(root)
    digest = hashlib.sha256()
    for key in ('X_tr', 'y_tr', 'X_val', 'y_val'):
        digest.update(data[key].detach().cpu().numpy().tobytes())
    for name in ('data.py', 'train.py', 'model.py', 'optimizer.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    current = dict(fingerprint=digest.hexdigest(), device=str(data['X_tr'].device),
                   torch=torch.__version__, platform=platform.platform(), threads=torch.get_num_threads(),
                   part3_sha256=sha(Path(__file__).with_name('part3.py')))
    stored = json.loads(lock.read_text())['provenance']
    if all(stored.get(k) == v for k, v in current.items()):
        return str(root)
    key = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()[:12]
    fresh = root.parent / f'reproduction_{key}'
    print('Different runtime/data/code: preserving sealed submission; output:', fresh)
    for folder in ('figures', 'results'):
        (fresh / folder).mkdir(parents=True, exist_ok=True)
    return str(fresh)


def write_report(root, results, p2, p3, baseline, final, baseline_scores, scores):
    by_id = {r['cfg']['exp_id']: r for r in results}
    base = baseline['summary']
    noise = p2['val_macro_f1']['noise_2sigma']
    plan = json.loads((root / 'results/part3_plan.json').read_text())
    health = json.loads((root / 'results/part1_checks.json').read_text())['summary']
    lines = [
        '# Báo cáo Lab Day 1 — MSSV 2A202602853\n',
        '## 1. Thiết lập\n',
        f"Forest CoverType, split metadata giữ nguyên: 464,809 train / 116,203 eval. Validation phân tầng 20%, seed 42: 371,847 train / 92,962 val. Chuẩn hóa 10 cột liên tục chỉ bằng train; 44 cột nhị phân giữ nguyên. PyTorch {p2['provenance']['torch']}, {p2['provenance']['device']}, {p2['provenance']['threads']} CPU threads. GPU memory không áp dụng trên CPU.\n",
        'M-base tự định nghĩa: 54→256→128→7, 47,879 tham số. Baseline CE, SGD momentum=0.9, LR=0.1, batch=512, He, FP32, 20 epoch, không dropout/clip/weight decay. Adam dùng betas=(0.9,0.999), eps=1e-8, weight_decay=0. Tất cả so sánh kỹ thuật dùng cùng seed 1 và 20 epoch; train loss đo eval trên 50,000 mẫu cố định.\n',
        '## 2. Kiểm tra ban đầu và độ nhiễu\n',
        f"`base-s1`: step0 CE={base['step0_loss']:.6f}, khác ln(7)=1.945910 vì logit He ngẫu nhiên không đều; không sửa He để ép loss. Part 1: shape (B,7), 47,879 tham số; tất cả 6 tensor tham số có gradient khác 0; overfit 20 mẫu, {health['steps']} bước, loss={health['overfit_loss']:.9f}, accuracy={health['overfit_acc']:.2%}. [Ảnh overfit](figures/part1_overfit20.png). Mốc majority validation={p2['majority_val_acc']:.6f}.\n",
        f"Baseline seeds 1/2/3: val accuracy={p2['val_acc']['mean']:.6f} ± {p2['val_acc']['std']:.6f}; macro-F1={p2['val_macro_f1']['mean']:.6f} ± {p2['val_macro_f1']['std']:.6f}; **2σ={noise:.6f}**. Đây là mốc nhiễu tham khảo từ 3 seed, không phải kiểm định ý nghĩa thống kê và không đại diện mọi cấu hình. [Ảnh baseline](figures/compare_baseline_seeds.png).\n",
        '## 3. Kết quả theo chủ đề\n',
        'Dự đoán dưới đây được ghi vào `results/part3_plan.json` trước khi chạy. Mỗi run có JSON, dòng Excel và ảnh riêng. Metric tại checkpoint có val loss thấp nhất; so CE/MSE bằng F1, không so loss khác thang. Đối chiếu từ một seed chỉ mô tả hướng quan sát, chưa chứng minh cơ chế nhân quả.\n',
    ]
    for group, title in [('loss', 'Hàm mất mát'), ('optimizer', 'Bộ tối ưu'), ('hparam', 'Batch size'),
                         ('dropout', 'Dropout'), ('clipping', 'Gradient clipping'), ('amp', 'Mixed precision'), ('init', 'Khởi tạo')]:
        lines.append(f'### {title}\n')
        for item in (i for i in plan if i['cfg']['group'] == group):
            r = by_id[item['cfg']['exp_id']]
            s, h = r['summary'], r['history']
            delta = s['val_macro_f1'] - base['val_macro_f1'] if s['val_macro_f1'] is not None else None
            effect = (f"ΔF1 so `base-s1`={delta:+.6f}, {'vượt' if abs(delta) > noise else 'chưa vượt'} 2σ baseline"
                      if delta is not None else 'Không có checkpoint hợp lệ; không xếp hạng chất lượng')
            exp_id = r['cfg']['exp_id']
            if s['diverged'] or not h['epoch']:
                matched = None
                observed = 'run không hoàn tất; không đủ evidence để đối chiếu chất lượng/timing'
            elif exp_id.startswith('opt-adam') and len(h['epoch']) >= 3:
                matched = h['val_macro_f1'][2] > baseline['history']['val_macro_f1'][2]
                observed = f"F1 epoch 3={h['val_macro_f1'][2]:.6f} vs baseline={baseline['history']['val_macro_f1'][2]:.6f}"
            elif exp_id == 'clip-high-none':
                rises = sum(b > a * 1.05 for a, b in zip(h['val_loss'], h['val_loss'][1:]))
                matched = s['diverged'] or rises > 0
                observed = f"{rises} lần val loss tăng >5%; diverged={s['diverged']}"
            elif exp_id == 'clip-high-on':
                ref = by_id['clip-high-none']['summary']
                matched = not s['diverged'] and (ref['diverged'] or s['val_macro_f1'] > ref['val_macro_f1'])
                observed = f"high-LR no-clip F1={ref['val_macro_f1']}, clipped F1={s['val_macro_f1']}"
            elif group == 'amp':
                amp_median = float(np.median(h['epoch_time_s']))
                base_median = float(np.median(baseline['history']['epoch_time_s']))
                matched = amp_median >= base_median
                observed = f"median time/epoch={amp_median:.3f}s vs FP32={base_median:.3f}s"
            elif exp_id == 'init-xavier':
                matched = s['initial_activation_std'][1] < base['initial_activation_std'][1]
                observed = 'std kích hoạt so với He trong bảng bên dưới'
            else:
                matched = delta is None or delta < 0
                observed = effect
            outcome = 'chưa kết luận' if matched is None else ('phù hợp hướng dự đoán' if matched else 'khác hướng dự đoán')
            verdict = f"Đối chiếu: {outcome} ({observed})."
            lines.append(f"- [`{exp_id}`](figures/{exp_id}.png): dự đoán **{item['hypothesis']}** Thực đo: F1={s['val_macro_f1']}, best epoch={s['best_epoch']}, diverged={s['diverged']}; {effect}. {verdict} {item['mechanism']}\n")
        lines.append(f'[Ảnh chồng {group}](figures/compare_{group}.png).\n')
        if group == 'optimizer':
            tuned = {}
            for r in [baseline] + [by_id[i['cfg']['exp_id']] for i in plan if i['cfg']['group'] == group]:
                opt = r['cfg']['optimizer']
                if not r['summary']['diverged'] and (opt not in tuned or r['summary']['val_macro_f1'] > tuned[opt]['summary']['val_macro_f1']):
                    tuned[opt] = r
            lines.append('So ở LR tốt nhất **trong các LR đã thử**, cùng budget 20 epoch:\n\n| Optimizer | exp_id | LR | val F1 | best epoch |\n|---|---|---:|---:|---:|\n')
            for opt, r in tuned.items():
                lines.append(f"| {opt} | {r['cfg']['exp_id']} | {r['cfg']['lr']} | {r['summary']['val_macro_f1']:.6f} | {r['summary']['best_epoch']} |\n")
            lines.append('\nAdamW không được thử; với weight_decay=0, Adam và AdamW có cùng cập nhật. Khi decay>0, AdamW tách decay khỏi moments.\n')
        elif group == 'hparam':
            if by_id['batch-2048']['summary']['time_per_epoch_s'] is None:
                lines.append('Batch-size run không hoàn tất epoch; không so timing.\n')
                continue
            lines.append(f"371,847 mẫu: batch 512 có {int(np.ceil(371847/512))} updates/epoch; batch 2048 có {int(np.ceil(371847/2048))}. Thời gian epoch lần lượt {base['time_per_epoch_s']:.3f}s / {by_id['batch-2048']['summary']['time_per_epoch_s']:.3f}s. Không thử scaling LR/warmup; chưa thể kết luận batch lớn kèm LR tối ưu.\n")
        elif group == 'dropout':
            s = by_id['drop-0.1']['summary']
            if s['final_val_loss'] is None or s['final_train_loss'] is None:
                lines.append('Dropout run không có epoch hoàn tất; không tính gap.\n')
                continue
            lines.append(f"Gap cuối baseline={base['final_val_loss']-base['final_train_loss']:.6f}, dropout={s['final_val_loss']-s['final_train_loss']:.6f}. Khoảng cách nhỏ và baseline còn cải thiện không chứng minh quá khớp; dropout chỉ nên dùng khi có evidence train giảm nhưng val tăng, không mặc định giúp.\n")
        elif group == 'clipping':
            for exp in ('clip-normal', 'clip-high-none', 'clip-high-on'):
                s = by_id[exp]['summary']
                lines.append(f"`{exp}`: clip={by_id[exp]['cfg']['clip_norm']}, {s['clipped_steps']}/{s['total_steps']} bước bị cắt ({s['clip_fraction']:.2%}), diverged={s['diverged']}.\n")
            lines.append('Cặp LR=1.0 chỉ khác clip; clipping giới hạn norm gradient, không giới hạn trực tiếp bước momentum hoặc khôi phục neuron ReLU đã chết. Không mặc định gọi đó là “cứu được”.\n')
        elif group == 'amp':
            s = by_id[p3['amp_exp_id']]['summary']
            precision = by_id[p3['amp_exp_id']]['cfg']['precision'].upper()
            if s['diverged'] or s['time_per_epoch_s'] is None:
                lines.append(f"{precision} run không có epoch hoàn tất; không thể so thời gian. FP16: {p3['fp16_status']}; BF16: {p3['bf16_status']}.\n")
                continue
            amp_times = by_id[p3['amp_exp_id']]['history']['epoch_time_s']
            amp_median = float(np.median(amp_times))
            base_median = float(np.median(baseline['history']['epoch_time_s']))
            ratio = base_median / amp_median
            outliers = [i + 1 for i, t in enumerate(amp_times) if t > 5 * amp_median]
            lines.append(f"Wall-clock mean: FP32 {base['time_per_epoch_s']:.3f}s/epoch; {precision} {s['time_per_epoch_s']:.3f}s/epoch. Median: FP32={base_median:.3f}s, {precision}={amp_median:.3f}s, ratio={ratio:.3f}×; {precision} {'nhanh hơn theo median quan sát' if ratio > 1 else 'không nhanh hơn theo median quan sát'}. Epoch outlier >5×median={outliers}; mean có thể chứa quãng gián đoạn thực thi, không diễn giải thành chi phí tính toán. GPU memory=null trên CPU. FP16: {p3['fp16_status']}; BF16: {p3['bf16_status']}. Không suy rộng CPU sang GPU hoặc coi median là benchmark kiểm soát nhiễu; timing gồm đánh giá FP32. FP16 exponent hẹp cần GradScaler; BF16 exponent 8 bit như FP32 thường không cần scaling.\n")
        elif group == 'init':
            lines.append('\n| init | step0 loss | Std sau từng Linear, 512 mẫu val |\n|---|---:|---|\n')
            for exp in ('base-s1', 'init-zeros', 'init-normal', 'init-xavier'):
                s = by_id[exp]['summary']
                lines.append(f"| {by_id[exp]['cfg']['init']} | {s['step0_loss']:.6f} | {', '.join(f'{v:.6f}' for v in s['initial_activation_std'])} |\n")
            lines.append('\nMạng chỉ có hai lớp ẩn; khác biệt này không chứng minh hành vi mạng 30 lớp trong slide. Zero chỉ học bias lớp ra, không học đặc trưng ẩn.\n')
    lines.extend([
        '\n![So sánh optimizer](figures/compare_optimizer.png)\n',
        '## 4. Đánh giá cuối trên tập eval\n',
        f"Khóa lựa chọn **trước eval**: ứng viên `{p3['candidate_exp_id']}` thắng theo val F1; giữ hyperparameters và tăng budget lên 40 epoch × 3 seed. Final val F1 mean±sample std={p3['final_f1_mean']:.6f}±{p3['final_f1_std']:.6f}; chọn một mô hình `{p3['final_exp_id']}` theo val F1, không ensemble. Cấu hình đầy đủ trong `results/final_selection.json`. [Ảnh final](figures/compare_final.png).\n",
        '\n| Cấu hình | Seed | val F1 | eval F1 | eval acc |\n|---|---:|---:|---:|---:|\n',
        f"| base-s1 | 1 | {base['val_macro_f1']:.6f} | {baseline_scores['macro_f1']:.6f} | {baseline_scores['accuracy']:.6f} |\n",
        f"| {final['cfg']['exp_id']} | {final['cfg']['seed']} | {final['summary']['val_macro_f1']:.6f} | {scores['macro_f1']:.6f} | {scores['accuracy']:.6f} |\n",
        f"\nĐiểm do `scripts/evaluate.py` tạo: 116,203 row_id hợp lệ. Δeval F1={scores['macro_f1']-baseline_scores['macro_f1']:+.6f}; mốc 2σ val={noise:.6f}, không có nhiều seed eval nên không gọi đây là kiểm định nhiễu eval. Final eval−val F1={scores['macro_f1']-final['summary']['val_macro_f1']:+.6f}. Không chỉnh cấu hình sau eval.\n",
        '### Phân tích lỗi theo lớp\n\n| Lớp (0-based) | Support | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|\n',
    ])
    for c in scores['per_class']:
        lines.append(f"| {c['cls']} | {c['support']} | {c['precision']:.6f} | {c['recall']:.6f} | {c['f1']:.6f} |\n")
    hardest = min(scores['per_class'], key=lambda c: c['f1'])
    cm = np.array(scores['confusion_matrix'])
    mistakes = cm[hardest['cls']].copy()
    mistakes[hardest['cls']] = 0
    confused = int(mistakes.argmax())
    lines.extend([
        f"\nLớp khó nhất={hardest['cls']}, F1={hardest['f1']:.6f}, support={hardest['support']}; nhầm nhiều nhất sang lớp {confused}: {mistakes[confused]} mẫu. [Ma trận nhầm lẫn](figures/confusion_eval.png), hàng=thật/cột=dự đoán. Dữ liệu mất cân bằng là cơ chế khả dĩ; sự giống nhau đặc trưng chỉ là giả thuyết, chưa phân tích feature để chứng minh. Có thể thử class-weighted CE và đánh giá trên val trong nghiên cứu tiếp theo.\n",
        '\n![Ma trận nhầm lẫn](figures/confusion_eval.png)\n',
        '## 5. Câu hỏi dẫn dắt\n',
        'Các câu về optimizer, dropout, clipping, precision và initialization được trả lời ở mục 3. Khi loss không giảm sau 2,000 bước, ba kiểm tra đầu tiên: (1) loss bước 0 so ln(C), kiểm tra nhãn/scale/logits và softmax hai lần; (2) tắt regularization, overfit 20 mẫu để phân biệt lỗi pipeline với thiếu khả năng tổng quát; (3) sau backward kiểm tra gradient từng tham số, zero_grad, optimizer và mức LR để phát hiện graph bị ngắt hoặc neuron chết.\n',
        '## 6. Hạn chế và điều bất ngờ\n',
        'Một seed cho ablation, ba seed cho baseline/final; 2σ baseline chỉ là heuristic. LR search hữu hạn; so optimizer kèm LR là ngoại lệ có chủ đích. MSE chưa được tối ưu LR riêng; hồi quy logit không kiểm chứng saturation của MSE trên softmax. Batch khác làm số update khác. Final tăng epoch nên không quy toàn bộ cải thiện cho một kỹ thuật. CPU không chứng minh AMP GPU; timing có quãng gián đoạn. Không nghiên cứu calibration hoặc feature similarity. Step0 He cao hơn ln(7) là kết quả ngoài giả định logit gần đều, đã chẩn đoán ở Part 1.\n',
        '## 7. Phụ lục\n',
        f"{len(results)} runs: `results/*.json`, `figures/<exp_id>.png` và ảnh chồng; `experiments.xlsx` giữ template/formulas/Seeds/Summary; `predictions_eval.csv`, `eval_result.json`, `REPORT.md`, `code/lab.ipynb` cùng toàn bộ mã trong `code/`. Không nộp dữ liệu gốc hoặc checkpoint. Tổng thời gian epoch đã đo={sum(sum(r['history']['epoch_time_s']) for r in results)/60:.1f} phút, chưa gồm plotting, nạp dữ liệu, Part 0–1 và rerun để lấy weights. Chạy lại: `python submission_2A202602853/code/run_part34.py` từ repo sau khi split_data.py.\n",
        'Khi runtime/data/code khác bản đã khóa, notebook tự chuyển output sang `reproduction_<hash>` trước khi ghi kết quả. Có thể tự đặt `LAB_OUT_DIR` thành một đường dẫn tuyệt đối tới thư mục output mới; không dùng lại thư mục đã chấm eval để đổi cấu hình. Colab/Kaggle chưa được kiểm chứng trực tiếp trong lần chạy CPU này.\n',
    ])
    report = '\n'.join(lines)
    # Paragraphs need blank separators, while Markdown table rows must be contiguous.
    report = re.sub(r'(?m)(^\|[^\n]*\n)\n(?=\|)', r'\1', report)
    (root / 'REPORT.md').write_text(report, encoding='utf-8')


def run_part4(data, out_dir, repo_dir, part2_results, p2, part3_results, p3):
    root, repo = Path(out_dir).resolve(), Path(repo_dir).resolve()
    baseline = next(r for r in part2_results if r['cfg']['exp_id'] == 'base-s1')
    final = next(r for r in part3_results if r['cfg']['exp_id'] == p3['final_exp_id'])
    selection = dict(baseline_cfg=baseline['cfg'], final_cfg=final['cfg'],
                     selection=p3['selection'], candidate_exp_id=p3['candidate_exp_id'],
                     provenance=p3['provenance'], evaluator_sha256=sha(repo / 'scripts/evaluate.py'),
                     metadata_sha256=sha(repo / 'data/split_metadata.csv'),
                     eval_npz_sha256=sha(repo / 'data/processed/eval.npz'),
                     raw_data_sha256=sha(repo / 'data/covtype.csv.gz'), eval_used_for_selection=False)
    lock = root / 'results/final_selection.json'
    previous = json.loads(lock.read_text()) if lock.exists() else None
    selection = json.loads(json.dumps(selection))
    if previous is not None and previous != selection:
        raise ValueError('Final selection already frozen; do not change configuration after seeing eval')
    lock.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    manifest_path = root / 'results/evaluation_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    training = {k: data[k] for k in ('X_tr', 'y_tr', 'X_val', 'y_val')}
    scores_by_id = {}
    for name, result, pred_path, score_path in [
        ('baseline', baseline, root / 'results/predictions_baseline.csv', root / 'results/baseline_eval_result.json'),
        ('final', final, root / 'predictions_eval.csv', root / 'eval_result.json'),
    ]:
        prior = manifest.get(name, {})
        reusable = (pred_path.exists() and score_path.exists()
                    and prior.get('selection_sha256') == sha(lock)
                    and prior.get('pred_sha256') == sha(pred_path)
                    and prior.get('scores_sha256') == sha(score_path))
        if not reusable:
            if result.get('best_state') is None:
                print('Retraining metrics-only result for weights:', result['cfg']['exp_id'])
                provenance = result['provenance']
                fresh = run_experiment(result['cfg'], training)
                fresh['provenance'] = provenance
                result.update(fresh)
                save_result(result, str(root / 'results'))
                plot_run(result, str(root / f"figures/{result['cfg']['exp_id']}.png"))
            final_eval(result['cfg'], result, data, str(pred_path))
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            proc = subprocess.run([sys.executable, str(repo / 'scripts/evaluate.py'),
                                   '--pred', str(pred_path), '--out', str(score_path)],
                                  cwd=repo, env=env, capture_output=True, text=True, encoding='utf-8', check=True)
            print(proc.stdout)
            manifest[name] = dict(selection_sha256=sha(lock), pred_sha256=sha(pred_path),
                                  scores_sha256=sha(score_path), exp_id=result['cfg']['exp_id'])
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        else:
            print('Reusing sealed official evaluation:', name)
        scores_by_id[result['cfg']['exp_id']] = json.loads(score_path.read_text())
    all_results = load_results(str(root / 'results'))
    # Baselines first; remaining runs sorted by exp_id for deterministic workbook order.
    all_results.sort(key=lambda r: (r['cfg']['group'] != 'baseline', r['cfg']['exp_id']))
    plan = {i['cfg']['exp_id']: i for i in json.loads((root / 'results/part3_plan.json').read_text())}
    rows = []
    for r in all_results:
        notes = f"CPU GPU-memory=N/A; {p3['fp16_status']}." if r['cfg']['group'] == 'amp' else 'GPU-memory=N/A on CPU.'
        if r['cfg']['exp_id'] in plan:
            notes += ' Prediction: ' + plan[r['cfg']['exp_id']]['hypothesis']
        if r['cfg']['loss'] == 'mse':
            notes += ' Raw-logit MSE vs one-hot; ln(7) step0-gap formula is only meaningful for CE.'
        if r['cfg']['group'] == 'clipping':
            notes += f" Clipped {r['summary']['clipped_steps']}/{r['summary']['total_steps']} steps."
        if r['cfg']['group'] == 'amp' and r['history']['epoch_time_s']:
            median = float(np.median(r['history']['epoch_time_s']))
            if max(r['history']['epoch_time_s']) > median * 5:
                notes += f' Wall-clock mean includes long execution interruption; median={median:.3f}s. See REPORT.'
        if r['cfg']['group'] == 'final':
            notes += f" Selected from {p3['candidate_exp_id']} by validation; extended 40 epochs."
        rows.append(to_row(r, scores_by_id.get(r['cfg']['exp_id']), notes))
    write_xlsx(rows, str(repo / 'templates/experiment_table_template.xlsx'), str(root / 'experiments.xlsx'))
    scores = scores_by_id[final['cfg']['exp_id']]
    cm = np.array(scores['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap='Blues')
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=9)
    ax.set(xlabel='Predicted class', ylabel='True class', title='Final eval confusion matrix',
           xticks=range(7), yticks=range(7))
    fig.tight_layout()
    fig.savefig(root / 'figures/confusion_eval.png', dpi=150)
    plt.close(fig)
    write_report(root, all_results, p2, p3, baseline, final,
                 scores_by_id[baseline['cfg']['exp_id']], scores)
    summary = dict(final_exp_id=final['cfg']['exp_id'], final_eval_macro_f1=scores['macro_f1'],
                   final_eval_accuracy=scores['accuracy'], baseline_eval=scores_by_id[baseline['cfg']['exp_id']],
                   experiment_count=len(rows), prediction_rows=scores['n_eval'],
                   metadata_sha256=sha(repo / 'data/split_metadata.csv'), selection_sha256=sha(lock))
    (root / 'results/part4_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary
