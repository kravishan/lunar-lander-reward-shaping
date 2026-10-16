import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

name = sys.argv[1] if len(sys.argv) > 1 else "ppo_lunar"
model = PPO.load(name, device="cpu")

env = gym.make("LunarLander-v3", render_mode="rgb_array")   # frames as images, no window
env = RecordVideo(env, video_folder="videos", name_prefix="baseline",
                  episode_trigger=lambda ep: True)          # record every episode

for ep in range(3):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
env.close()
print("Saved to the videos folder")