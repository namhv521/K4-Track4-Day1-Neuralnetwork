"""Run notebook setup and Part 2, preserving existing Part 0–1 outputs."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import torch

from data import prepare_data
from part2 import baseline_notes


def main():
    repo = Path(__file__).resolve().parents[2]
    path = Path(__file__).with_name('lab.ipynb')
    metadata = repo / 'data/split_metadata.csv'
    before = hashlib.sha256(metadata.read_bytes()).hexdigest()
    notebook = json.loads(path.read_text(encoding='utf-8'))
    os.chdir(repo)
    # Small GEMMs on CPU are faster with a few threads than the machine default.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ns = {'__name__': '__main__'}
    exec(''.join(notebook['cells'][1]['source']), ns)
    ns['data'] = prepare_data(ns['device'], seed=42,
                              processed_dir=str(repo / 'data/processed'))
    class Tee(io.StringIO):
        def write(self, value):
            import sys
            sys.__stdout__.write(value)
            sys.__stdout__.flush()
            return super().write(value)
    captured = Tee()
    with contextlib.redirect_stdout(captured):
        exec(''.join(notebook['cells'][9]['source']), ns)
    notebook['cells'][9]['execution_count'] = 4
    notebook['cells'][9]['outputs'] = [dict(output_type='stream', name='stdout',
                                          text=captured.getvalue().splitlines(keepends=True))]
    notebook['cells'][10]['source'] = baseline_notes(ns['part2_results'], ns['part2_summary'])
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == before
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Saved actual Part 2 notebook output; preserved Part 0–1; stopped before Part 3.')


if __name__ == '__main__':
    main()
