"""
Make per-step polynomial action labels for PushT, then verify them by replaying every demo.

Usage:
    python make_perstep_labels.py --budget 6
    python make_perstep_labels.py --budget 2 --no-verify

Writes <src>_perstep_b<budget>.zarr next to the source: identical to the source except data/action,
which holds the per-step labels (2*budget numbers per step). The original PD targets are kept as
data/action_pd. See diffusion_policy/common/perstep_motion.py for the label definition.
"""
import pathlib
import shutil
import click
import numpy as np
import zarr
from diffusion_policy.common.perstep_motion import (
    StepDecoder, labels_from_dense, pd_replay, pd_target, TICKS)


def replay(states, label_fn_or_targets, mode):
    """Replay one demo in PushTEnv. mode='pd': recorded targets; mode='labels': per-step executor."""
    from diffusion_policy.env.pusht.pusht_env import PushTEnv
    from pymunk.vec2d import Vec2d
    env = PushTEnv(legacy=False, reset_to_state=states[0])
    env.reset()
    out, rewards = [states[0]], []
    if mode == 'pd':
        for u in label_fn_or_targets[:-1]:
            obs, r, _, _ = env.step(u)
            out.append(obs); rewards.append(r)
    else:
        labels, decoder = label_fn_or_targets
        x0, v0, dt = np.array(env.agent.position), np.zeros(2), 1.0 / env.sim_hz
        for lab in labels[:-1]:
            for x_ref in decoder.ticks(x0, v0, lab):
                x, v = np.array(env.agent.position), np.array(env.agent.velocity)
                u = pd_target(x, v, x_ref, env.k_p, env.k_v, dt)
                env.agent.velocity = Vec2d(*(v + (env.k_p * (u - x) - env.k_v * v) * dt))
                env.space.step(dt)
            x0, v0 = decoder.end_state(lab)
            obs, r, _, _ = env.step(None)
            out.append(obs); rewards.append(r)
    out = np.array(out)
    block_err = np.linalg.norm(out[:, 2:4] - states[:, 2:4], axis=1).max()
    return block_err, max(rewards)


@click.command()
@click.option('--budget', type=int, required=True, help='numbers per axis per 0.1 s step (2, 4, 6, ...)')
@click.option('--src', default='/home/elijahong/robot-learning/data/pusht/pusht_cchi_v7_replay.zarr')
@click.option('--dst', default=None, help='default: <src>_perstep_b<budget>.zarr')
@click.option('--verify/--no-verify', default=True, help='replay all demos with the labels (~2 min)')
@click.option('--overwrite', is_flag=True)
def main(budget, src, dst, verify, overwrite):
    src = pathlib.Path(src)
    dst = pathlib.Path(dst) if dst else src.with_name(f'{src.stem}_perstep_b{budget}.zarr')
    zs = zarr.open(str(src), 'r')
    ends = zs['meta/episode_ends'][:]
    starts = np.r_[0, ends[:-1]]
    U = zs['data/action'][:].astype(np.float64)
    S = zs['data/state'][:].astype(np.float64)

    decoder = StepDecoder(budget)
    labels = np.zeros((len(U), 2 * budget), np.float32)
    worst_replay, worst_decode = 0.0, 0.0
    for s0, e0 in zip(starts, ends):
        xt = pd_replay(U[s0:e0], S[s0, :2])
        worst_replay = max(worst_replay, np.abs(xt[::TICKS] - S[s0:e0, :2]).max())
        lab = labels_from_dense(xt, budget)
        labels[s0:e0] = lab
        # decoding the labels must reproduce the fitted pieces' end points exactly
        x0, v0 = xt[0], np.zeros(2)
        for k in range(e0 - s0 - 1):
            ticks = decoder.ticks(x0, v0, lab[k])
            worst_decode = max(worst_decode, np.abs(ticks[-1] - xt[(k + 1) * TICKS]).max())
            x0, v0 = decoder.end_state(lab[k])
    print(f'[labels] budget={budget}: {len(ends)} episodes, {len(U)} steps, action dim {2 * budget}')
    print(f'[check] PD replay vs recorded agent positions: max {worst_replay:.1e} px (must be ~0)')
    print(f'[check] decoded step ends vs dense motion:     max {worst_decode:.1e} px (must be ~0)')

    if dst.exists():
        if not overwrite:
            raise click.ClickException(f'{dst} exists (use --overwrite)')
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    zd = zarr.open(str(dst), 'r+')
    zd['data'].create_dataset('action_pd', data=zs['data/action'][:], chunks=zs['data/action'].chunks)
    del zd['data/action']
    zd['data'].create_dataset('action', data=labels, chunks=(min(len(labels), 4096), 2 * budget))
    zd['data/action'].attrs['perstep_budget'] = budget
    print(f'[write] {dst}')

    if verify:
        pd_res = [replay(S[s0:e0], U[s0:e0], 'pd') for s0, e0 in zip(starts, ends)]
        lb_res = [replay(S[s0:e0], (labels[s0:e0], decoder), 'labels') for s0, e0 in zip(starts, ends)]
        b = np.array([r[0] for r in lb_res])
        print(f'[verify] demos replayed with their own PD targets: mean best reward {np.mean([r[1] for r in pd_res]):.4f}')
        print(f'[verify] demos replayed from budget-{budget} labels: mean best reward {np.mean([r[1] for r in lb_res]):.4f}, '
              f'block error median {np.median(b):.2f} px / max {b.max():.1f} px, within 1 px {np.mean(b < 1) * 100:.0f}%')


if __name__ == '__main__':
    main()
