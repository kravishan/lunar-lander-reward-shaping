import argparse
import time
from stable_baselines3 import PPO
from rewards import TimePenalty, UprightBonus, BonusLogger
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def main():
    # Read settings from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")             # "cpu" or "cuda"
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--name", default="ppo_lunar")
    parser.add_argument("--reward", default="none")          # "none" or "time"
    parser.add_argument("--penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 16 copies of the game, each in its own process (uses your CPU cores)
    env = make_vec_env("LunarLander-v3", n_envs=16, vec_env_cls=SubprocVecEnv)

    if args.reward == "time":
        wrapper, wrapper_kwargs = TimePenalty, {"penalty": args.penalty}
    elif args.reward == "hover":
        wrapper, wrapper_kwargs = UprightBonus, {"bonus": 1.0}
    else:
        wrapper, wrapper_kwargs = None, None

    env = make_vec_env("LunarLander-v3", n_envs=16, vec_env_cls=SubprocVecEnv,
                       wrapper_class=wrapper, wrapper_kwargs=wrapper_kwargs,
                       seed=args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        device=args.device,
        tensorboard_log="logs",   # saves the learning curve
        verbose=1,
    )

    start = time.time()
    model.learn(total_timesteps=args.steps, tb_log_name=args.name, callback=BonusLogger())
    minutes = (time.time() - start) / 60
    print(f"\n{args.device}: {args.steps:,} steps in {minutes:.1f} minutes")

    model.save(args.name)
    env.close()


if __name__ == "__main__":   # required on Windows when using SubprocVecEnv
    main()