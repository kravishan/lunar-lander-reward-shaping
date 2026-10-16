import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

model = PPO.load("ppo_lunar", device="cpu")

# 1. Proper test: 20 fresh episodes, no window
eval_env = Monitor(gym.make("LunarLander-v3"))
mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(f"Test score over 20 episodes: {mean:.1f} +/- {std:.1f}\n")

# 2. Watch 5 episodes
env = gym.make("LunarLander-v3", render_mode="human")
for ep in range(5):
    obs, info = env.reset()
    total, steps, done = 0.0, 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        done = terminated or truncated

    if truncated:
        outcome = "TIMED OUT (hovered 1000 frames)"
    elif reward == 100:
        outcome = "LANDED"
    else:
        outcome = "CRASHED or flew off"
    print(f"Episode {ep + 1}: {outcome}, score {total:.1f}, {steps} frames")
env.close()