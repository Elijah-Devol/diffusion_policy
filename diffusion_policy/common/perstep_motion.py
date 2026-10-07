"""
Per-step polynomial actions for PushT.

budget b = numbers per axis per 0.1 s action step (b >= 2).
Each step is one polynomial piece of degree b+1 in tau = (t - t_step) / 0.1, tau in [0, 1]:
  start (inherited, not predicted): position and velocity where the previous step ended
  label (predicted, 2*b numbers):   positions at tau = 1/(b-1), 2/(b-1), ..., 1   and   velocity at tau = 1
  layout: [p1_x, p1_y, ..., p(b-1)_x, p(b-1)_y, v_x, v_y]
b=2 -> cubic per step (end position + end velocity)
b=4 -> quintic per step (positions at 1/3, 2/3, 1 + end velocity)
b=6 -> 7th order per step (positions at 0.2, 0.4, 0.6, 0.8, 1 + end velocity)

The piece is converted to PushT actions by sampling it at the 10 physics ticks of the step and,
at every tick, sending the PD target that lands the agent exactly on the next sample.
"""
import numpy as np

STEP_DT = 0.1            # one action step (10 Hz)
TICKS = 10               # physics ticks per action step (PushTEnv: sim_hz=100, control_hz=10)
TICK_DT = STEP_DT / TICKS
KP, KV = 100.0, 20.0     # PushTEnv PD gains (k_p, k_v)


def label_taus(budget):
    assert budget >= 2, 'budget must be >= 2'
    return np.arange(1, budget) / (budget - 1)


def stay_label(x, budget):
    """Label for 'hold position x': all positions at x, zero end velocity."""
    x = np.asarray(x, dtype=np.float64)
    return np.concatenate([np.tile(x, budget - 1), np.zeros(2)]).astype(np.float32)


def pd_target(x, v, x_next, kp=KP, kv=KV, dt=TICK_DT):
    """PD target that moves PushT's (kinematic) agent from (x, v) exactly to x_next in one physics tick."""
    return x + ((x_next - x) / dt - v * (1.0 - kv * dt)) / (kp * dt)


def pd_replay(targets, x0, kp=KP, kv=KV):
    """Exact 100 Hz agent positions produced by PushTEnv's PD for recorded targets (agent starts at rest)."""
    x, v = np.asarray(x0, np.float64).copy(), np.zeros(2)
    xs = [x.copy()]
    for u in np.asarray(targets, np.float64)[:-1]:
        for _ in range(TICKS):
            v = v + (kp * (u - x) - kv * v) * TICK_DT
            x = x + v * TICK_DT
            xs.append(x.copy())
    return np.array(xs)


class StepDecoder:
    """Rebuilds one step's polynomial from (start position, start velocity, label) -- lossless."""

    def __init__(self, budget):
        self.budget = budget
        taus = label_taus(budget)
        n = budget + 2                                            # coefficients = conditions
        rows = [np.eye(n)[0], np.eye(n)[1]]                       # p(0), p'(0)
        rows += [taus_j ** np.arange(n) for taus_j in taus]       # p(tau_j)
        rows += [np.r_[0.0, np.arange(1, n, dtype=np.float64)]]   # p'(1)
        self.Minv = np.linalg.inv(np.array(rows))
        t = np.arange(1, TICKS + 1) / TICKS
        self.T = t[:, None] ** np.arange(n)[None]                 # monomials at the 10 ticks

    def split(self, label):
        label = np.asarray(label, np.float64)
        return label[:-2].reshape(self.budget - 1, 2), label[-2:]

    def coefficients(self, x0, v0, label):
        p, v1 = self.split(label)
        rhs = np.vstack([x0, STEP_DT * np.asarray(v0, np.float64), p, STEP_DT * v1])
        return self.Minv @ rhs                                    # (budget+2, 2)

    def ticks(self, x0, v0, label):
        """Reference positions at the 10 physics ticks of the step (tick 1 ... tick 10 = end of step)."""
        return self.T @ self.coefficients(x0, v0, label)

    def end_state(self, label):
        p, v1 = self.split(label)
        return p[-1].copy(), v1.copy()


def knot_velocities(xt):
    """Velocity at every step boundary of a dense (100 Hz) motion: central difference, rest at start."""
    xk = xt[::TICKS]
    v = np.zeros_like(xk)
    L = len(xk)
    if L > 2:
        v[1:-1] = (xt[TICKS + 1::TICKS][:L - 2] - xt[TICKS - 1::TICKS][:L - 2]) / (2 * TICK_DT)
    if L > 1:
        v[-1] = (xt[-1] - xt[-2]) / TICK_DT
    return xk, v


def labels_from_dense(xt, budget):
    """
    Fit one piece per step to a dense motion (positions at every tick) and return labels (L, 2*budget).
    Position and velocity are shared at step boundaries; the remaining (budget-2) shape parameters of
    each piece are least-squares fitted to the step's interior ticks. The last label holds position.
    """
    xk, v = knot_velocities(xt)
    L, m, taus = len(xk), budget - 2, label_taus(budget)
    tau = np.arange(TICKS + 1) / TICKS
    H3 = np.stack([2 * tau**3 - 3 * tau**2 + 1, tau**3 - 2 * tau**2 + tau,
                   -2 * tau**3 + 3 * tau**2, tau**3 - tau**2], 1)
    W = (tau**2 * (1 - tau)**2)[:, None] * tau[:, None] ** np.arange(m)[None]
    Wpinv = np.linalg.pinv(W[1:-1]) if m > 0 else None
    tj = taus
    H3j = np.stack([2 * tj**3 - 3 * tj**2 + 1, tj**3 - 2 * tj**2 + tj, -2 * tj**3 + 3 * tj**2, tj**3 - tj**2], 1)
    Wj = (tj**2 * (1 - tj)**2)[:, None] * tj[:, None] ** np.arange(m)[None]
    out = np.zeros((L, 2 * budget), np.float32)
    for k in range(L - 1):
        ends = np.stack([xk[k], STEP_DT * v[k], xk[k + 1], STEP_DT * v[k + 1]])
        seg = xt[k * TICKS:(k + 1) * TICKS + 1]
        c = Wpinv @ (seg[1:-1] - (H3 @ ends)[1:-1]) if m > 0 else np.zeros((0, 2))
        p = H3j @ ends + (Wj @ c if m > 0 else 0.0)
        out[k] = np.concatenate([p.ravel(), v[k + 1]])
    out[L - 1] = stay_label(xk[L - 1], budget)
    return out
