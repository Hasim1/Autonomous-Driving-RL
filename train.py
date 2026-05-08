import os

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import HALF_TIMESTEPS, MODEL_PATHS, N_ENVS, SEED, TOTAL_TIMESTEPS
from environment import create_environment
from model import build_ppo_model, get_training_device
from utils import (
    RewardLoggerCallback,
    now,
    plot_rewards,
    prepare_directories,
    save_rewards,
    save_training_summary,
)

# Prevent PyTorch CPU thread oversubscription, which can make small RL models very slow.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)


def make_env(rank: int):
    """Create one monitored environment for vectorized PPO training."""

    def _init():
        env = create_environment(render_mode=None)
        env.reset(seed=SEED + rank)
        return Monitor(env)

    return _init


def train_agent() -> None:
    """Train PPO and save untrained, half-trained, and fully-trained checkpoints."""
    prepare_directories()
    start_time = now()

    env = DummyVecEnv([make_env(rank) for rank in range(N_ENVS)])
    model = build_ppo_model(env)
    reward_logger = RewardLoggerCallback()

    model.save(MODEL_PATHS["untrained"])
    print(f"Saved untrained model: {MODEL_PATHS['untrained']}")

    print(f"Training half model for {HALF_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=HALF_TIMESTEPS, callback=reward_logger)
    model.save(MODEL_PATHS["half_trained"])
    print(f"Saved half-trained model: {MODEL_PATHS['half_trained']}")

    remaining_timesteps = TOTAL_TIMESTEPS - HALF_TIMESTEPS
    print(f"Training full model for another {remaining_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=reward_logger,
        reset_num_timesteps=False,
    )
    model.save(MODEL_PATHS["fully_trained"])
    print(f"Saved fully-trained model: {MODEL_PATHS['fully_trained']}")

    elapsed_seconds = now() - start_time
    rewards = reward_logger.episode_rewards
    save_rewards(rewards)
    plot_rewards(rewards)
    save_training_summary(
        device=get_training_device(),
        total_timesteps=TOTAL_TIMESTEPS,
        elapsed_seconds=elapsed_seconds,
        rewards=rewards,
    )

    print("Saved graph/team files:")
    print("- logs/episode_rewards.csv")
    print("- logs/training_summary.txt")
    print("- assets/reward_plot.png")
    print(f"Finished in {elapsed_seconds:.2f} seconds.")

    env.close()


if __name__ == "__main__":
    train_agent()
