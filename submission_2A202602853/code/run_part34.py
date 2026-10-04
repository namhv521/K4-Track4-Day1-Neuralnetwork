"""Execute every notebook code cell and save real outputs after each completed cell."""
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import torch


def main():
    repo = Path(__file__).resolve().parents[2]
    path = Path(__file__).with_name('lab.ipynb')
    metadata = repo / 'data/split_metadata.csv'
    before = hashlib.sha256(metadata.read_bytes()).hexdigest()
    notebook = json.loads(path.read_text(encoding='utf-8'))
    os.chdir(repo)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    ns = {'__name__': '__main__'}

    class Tee(io.StringIO):
        def write(self, value):
            sys.__stdout__.write(value)
            sys.__stdout__.flush()
            return super().write(value)

    count = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        captured = Tee()
        with contextlib.redirect_stdout(captured):
            exec(compile(''.join(cell['source']), f'lab.ipynb:cell{index}', 'exec'), ns)
        cell['execution_count'] = count
        cell['outputs'] = [dict(output_type='stream', name='stdout', text=captured.getvalue().splitlines(keepends=True))]
        if index == 9:
            from part2 import baseline_notes
            notebook['cells'][10]['source'] = baseline_notes(ns['part2_results'], ns['part2_summary'])
        assert hashlib.sha256(metadata.read_bytes()).hexdigest() == before
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('PASS: all notebook code cells executed; split metadata unchanged.')


if __name__ == '__main__':
    main()
