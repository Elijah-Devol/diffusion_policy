import gym
import numpy as np
from gym import spaces
from pymunk.vec2d import Vec2d
from diffusion_policy.common.perstep_motion import StepDecoder, pd_target, TICKS


class PerStepExecutor(gym.Wrapper):
    """
    Executes one per-step polynomial label (2*budget numbers) as one PushT step:
    rebuild the step's curve from where the previous step ended, sample it at the 10 physics ticks,
    and at every tick send PushT's PD the target that lands the agent exactly on the next sample.
    Wrap the raw PushT env with this (innermost), so MultiStepWrapper calls it once per step.
    """
    def __init__(self, env, budget):
        super().__init__(env)
        self.budget = budget
        self.decoder = StepDecoder(budget)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * budget,), dtype=np.float64)
        self.x0 = None
        self.v0 = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.x0 = np.array(self.env.unwrapped.agent.position, dtype=np.float64)
        self.v0 = np.zeros(2)
        return obs

    def step(self, label):
        base = self.env.unwrapped
        assert base.sim_hz // base.control_hz == TICKS
        dt = 1.0 / base.sim_hz
        label = np.asarray(label, dtype=np.float64)
        for x_ref in self.decoder.ticks(self.x0, self.v0, label):
            x = np.array(base.agent.position)
            v = np.array(base.agent.velocity)
            u = pd_target(x, v, x_ref, base.k_p, base.k_v, dt)               # the PD action for this tick
            base.agent.velocity = Vec2d(*(v + (base.k_p * (u - x) - base.k_v * v) * dt))  # PushT's PD law
            base.space.step(dt)
        self.x0, self.v0 = self.decoder.end_state(label)
        base.latest_action = self.x0.copy()                                    # marker in rendered frames
        return self.env.step(None)                                             # reward/obs, no extra motion
