"""Execute only notebook setup and Part 0, saving their actual outputs.

Run from repo root: python submission_2A202602853/code/run_part0.py
Part 1–4 cells are deliberately not executed.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path


def main(stop_heading="## Part 1", expected_cells=2):
    code_dir = Path(__file__).resolve().parent
    repo = code_dir.parent.parent
    notebook_path = code_dir / "lab.ipynb"
    metadata = repo / "data" / "split_metadata.csv"
    original_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    # Save figures without requiring a native GUI on local/headless runtimes.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(repo)
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown" and stop_heading in "".join(cell["source"]):
            break
        if cell["cell_type"] != "code":
            continue
        count += 1
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile("".join(cell["source"]), str(notebook_path), "exec"), namespace)
        output = captured.getvalue()
        cell["execution_count"] = count
        cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                            "text": output.splitlines(keepends=True)}]
        print(output, end="")
    assert count == expected_cells, "Unexpected number of executed cells"
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == original_hash
    if "part1_summary" in namespace:
        summary = namespace["part1_summary"]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown" and "**Nhận xét Part 1" in "".join(cell["source"]):
                cell["source"] = [
                    "**Nhận xét Part 1 (kết quả thực đo):**\n",
                    f"- M-base có {summary['params']:,} tham số; logits `(8,7)`.\n",
                    f"- He init: CE trên toàn bộ val trước cập nhật = {summary['step0_loss']:.6f}, so với ln(7) = {summary['ln7']:.6f}. Giá trị He lệch mốc, chưa phải phân phối dự đoán đồng đều.\n",
                    f"- Chẩn đoán theo GUIDE: logits ×0.1 cho CE = {summary['scaled_logits_ce']:.6f}; logits bằng 0 cho CE = {summary['uniform_ce']:.6f}. Kết quả phù hợp với nguyên nhân thang logits ngẫu nhiên của He. Phép giảm thang chỉ là đối chứng, không sửa khởi tạo He của baseline.\n",
                    f"- Model riêng quá khớp 20 mẫu sau {summary['steps']} bước: loss = {summary['overfit_loss']:.6f}, accuracy = {summary['overfit_acc']:.1%}.\n",
                    "- Cả 6 tensor tham số có gradient hữu hạn, norm khác 0 trước huấn luyện. Không dùng eval để kiểm tra model hoặc điều chỉnh.\n",
                    "- Thống kê kích hoạt đo sau mỗi Linear. Model thử 20 mẫu được tách riêng; Part 2 phải khởi tạo baseline mới.\n",
                    "\n![Quá khớp 20 mẫu](../figures/part1_overfit20.png)\n",
                ]
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Metadata unchanged (SHA256): {original_hash}")
    print(f"Saved {count} executed cells; stopped before {stop_heading}.")


if __name__ == "__main__":
    main()
