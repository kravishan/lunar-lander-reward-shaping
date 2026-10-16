import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


class TimePenalty(gym.Wrapper):
    """Charges a small fee every frame, so taking longer costs points."""

    def __init__(self, env, penalty=0.05):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward - self.penalty
        return obs, reward, terminated, truncated, info


class UprightBonus(gym.Wrapper):
    """+1 every frame the rocket is upright and slow. The broken reward."""

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus
        self.bonus_total = 0.0

    def reset(self, **kwargs):
        self.bonus_total = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs
        speed = (vx ** 2 + vy ** 2) ** 0.5

        extra = self.bonus if (abs(angle) < 0.2 and speed < 0.3) else 0.0
        self.bonus_total += extra

        if terminated or truncated:
            info["bonus_total"] = self.bonus_total
        return obs, reward + extra, terminated, truncated, info


class BonusLogger(BaseCallback):
    """Sends each episode's bonus total to the log and TensorBoard."""

    def _on_step(self):
        for info in self.locals["infos"]:
            if "bonus_total" in info:
                self.logger.record_mean("rollout/bonus_mean", info["bonus_total"])
        return True