"""results_table.py — Lưu JSON và điền bảng tổng hợp từ mẫu của bài lab.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def save_result(result, results_dir='../results'):
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{result['cfg']['exp_id']}.json"
    keys = ('cfg', 'history', 'summary', 'provenance')
    path.write_text(json.dumps({k: result[k] for k in keys if k in result},
                               ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    results = []
    for path in Path(results_dir).glob('*.json'):
        result = json.loads(path.read_text(encoding='utf-8'))
        if (all(key in result for key in ('cfg', 'summary', 'history'))
                and all(key in result['cfg'] for key in ('exp_id', 'group'))):
            results.append(result)
    return sorted(results, key=lambda r: r['cfg']['exp_id'])


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg = result['cfg']
    row = {**cfg, **result['summary'], 'hidden': '-'.join(map(str, cfg['hidden'])),
           'clip_norm': cfg['clip_norm'] if cfg['clip_norm'] is not None else 'none',
           'figure_file': f"figures/{cfg['exp_id']}.png", 'notes': notes}
    if eval_scores is not None:
        if cfg['group'] not in ('baseline', 'final'):
            raise ValueError('Eval scores are only allowed for baseline and final runs')
        row.update(eval_acc=eval_scores['accuracy'], eval_macro_f1=eval_scores['macro_f1'])
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    from copy import copy
    import openpyxl
    from openpyxl.formula.translate import Translator
    from openpyxl.workbook.properties import CalcProperties

    wb = openpyxl.load_workbook(template_path)
    ws = wb['Experiments']
    headers = [cell.value for cell in ws[1]]
    formula_columns = {'step0_gap_vs_lnC', 'gap_val_minus_train',
                       'delta_val_f1_vs_base', 'beyond_noise'}
    count = max(ws.max_row, len(rows) + 1)
    for index in range(2, count + 1):
        row = rows[index - 2] if index - 2 < len(rows) else {}
        for col, key in enumerate(headers, 1):
            cell = ws.cell(index, col)
            if index > 61:
                cell._style = copy(ws.cell(2, col)._style)
            if key in formula_columns:
                if cell.value is None:
                    origin = ws.cell(2, col)
                    cell.value = Translator(origin.value, origin=origin.coordinate).translate_formula(cell.coordinate)
            else:
                cell.value = row.get(key)
        if row:
            wrapped_lines = 1
            for col in (3, 28, 29):
                cell = ws.cell(index, col)
                alignment = copy(cell.alignment)
                alignment.wrap_text = True
                alignment.vertical = 'top'
                cell.alignment = alignment
                width = ws.column_dimensions[cell.column_letter].width
                wrapped_lines = max(wrapped_lines, math.ceil(len(str(cell.value or '')) / max(width - 2, 1)))
            ws.row_dimensions[index].height = max(24, wrapped_lines * 15)
    ws.auto_filter.ref = f'A1:AG{len(rows) + 1}'
    # Put baseline first so template comparisons and Seeds remain easy to inspect.
    seeds = [r['exp_id'] for r in rows if r['group'] == 'baseline']
    if len(seeds) > 5:
        raise ValueError('Template supports up to five baseline seeds')
    for index in range(2, 7):
        wb['Seeds'].cell(index, 1, seeds[index - 2] if index - 2 < len(seeds) else None)
        if index - 2 >= len(seeds):
            wb['Seeds'].cell(index, 1).value = None
    if count > 61:
        for sheet in (wb['Seeds'], wb['Summary']):
            for line in sheet:
                for cell in line:
                    if cell.data_type == 'f':
                        cell.value = cell.value.replace('$61', f'${count}')
    summary = wb['Summary']
    # The template has a spare row for the existing Part 2 LR-screening group.
    summary.cell(12, 1, 'lr_screen')
    for col in (1, 7, 8):
        summary.cell(12, col)._style = copy(summary.cell(11, col)._style)
    for col in range(2, 7):
        origin = summary.cell(11, col)
        summary.cell(12, col).value = Translator(origin.value, origin=origin.coordinate).translate_formula(summary.cell(12, col).coordinate)
        summary.cell(12, col)._style = copy(origin._style)
    summary.cell(12, 7, '—')
    for line in wb['Summary'].iter_rows(min_row=2):
        group = line[0].value
        matches = [r for r in rows if r['group'] == group]
        if matches:
            best = max((r for r in matches if r.get('val_macro_f1') is not None),
                       key=lambda r: r['val_macro_f1'], default=None)
            line[7].value = (f"Best val F1: {best['exp_id']} = {best['val_macro_f1']:.6f}. "
                             + best.get('notes', '')) if best else matches[0].get('notes', '')
            alignment = copy(line[7].alignment)
            alignment.wrap_text = True
            line[7].alignment = alignment
            summary.row_dimensions[line[7].row].height = max(30, math.ceil(len(line[7].value)/50) * 15)
    wb.calculation = CalcProperties(calcId=0, fullCalcOnLoad=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(out_path)
    except PermissionError:
        # Excel may hold the file open. Reuse it only if every cell agrees;
        # otherwise propagate the lock error instead of silently losing updates.
        if not Path(out_path).is_file():
            raise
        existing = openpyxl.load_workbook(out_path)
        try:
            if existing.sheetnames != wb.sheetnames:
                raise
            for sheet in wb:
                old = existing[sheet.title]
                if old.max_row != sheet.max_row or old.max_column != sheet.max_column:
                    raise
                for row in sheet:
                    for cell in row:
                        actual, expected = old[cell.coordinate].value, cell.value
                        # Excel/openpyxl serialize an empty text cell as blank.
                        actual = None if actual == '' else actual
                        expected = None if expected == '' else expected
                        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                            equal = math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
                        else:
                            equal = actual == expected
                        if not equal:
                            raise
        finally:
            existing.close()
        print('Reusing locked workbook: all cell values and formulas match', out_path)
