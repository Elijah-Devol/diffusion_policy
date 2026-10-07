"""
Compare in-training validation (test/mean_score every rollout_every epochs) across runs.

Usage:
    python compare_runs.py data/outputs/dp-04_pusht_image_seed42 data/outputs/dp-perstep-b*_seed42 --max_epoch 1000
"""
import json
import pathlib
import click
import numpy as np


def load(run_dir, max_epoch):
    evals = []
    with open(pathlib.Path(run_dir) / 'logs.json.txt') as f:
        for line in f:
            if '"test/mean_score"' not in line:
                continue
            r = json.loads(line)
            if max_epoch is None or r['epoch'] <= max_epoch:
                evals.append((r['epoch'], r['test/mean_score']))
    return evals


@click.command()
@click.argument('run_dirs', nargs=-1, required=True)
@click.option('--max_epoch', default=None, type=int, help='only count evaluations up to this epoch')
@click.option('--last', default=10, type=int, help='average of the last N evaluations')
def main(run_dirs, max_epoch, last):
    print(f"{'run':40s} {'evals':>5s} {'last epoch':>10s} {'best (epoch)':>15s} {f'mean last {last}':>12s} {'mean all':>9s}")
    for d in run_dirs:
        ev = load(d, max_epoch)
        if not ev:
            print(f'{pathlib.Path(d).name:40s}  no evaluations yet'); continue
        ep, sc = np.array([e for e, _ in ev]), np.array([s for _, s in ev])
        i = int(sc.argmax())
        print(f'{pathlib.Path(d).name:40s} {len(ev):5d} {ep[-1]:10d} {sc[i]:9.3f} ({ep[i]:4d}) '
              f'{sc[-last:].mean():12.3f} {sc.mean():9.3f}')


if __name__ == '__main__':
    main()
