import gymnasium as gym


class TimePenalty(gym.Wrapper):
    """Charges a small fee every frame, so taking longer costs points."""

    def __init__(self, env, penalty=0.05):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward - self.penalty
        return obs, reward, terminated, truncated, info