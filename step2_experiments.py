import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for episode in range(3):
    obs, info = env.reset()
    total_reward = 0.0
    step = 0
    done = False

    while not done:
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        if step % 20 == 0:
            print(f"step {step:4d} | action {action} | reward {reward:7.2f} | obs {obs.round(2)}")

        done = terminated or truncated

    print(f"Episode {episode + 1}: {step} steps, last reward {reward:.1f}, total score {total_reward:.1f}\n")
env.close()