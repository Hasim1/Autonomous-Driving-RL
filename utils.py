import csv
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from config import ASSET_DIR, LOG_DIR, MODEL_DIR, REWARD_LOG_PATH, REWARD_PLOT_PATH, SUMMARY_PATH


class RewardLoggerCallback(BaseCallback):
    """Record episode rewards during training."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True


def prepare_directories() -> None:
    """Create required output directories."""
    for directory in (MODEL_DIR, ASSET_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_rewards(rewards: Iterable[float], path: Path = REWARD_LOG_PATH) -> None:
    """Save episode rewards to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward"])
        for episode, reward in enumerate(rewards, start=1):
            writer.writerow([episode, reward])


def plot_rewards(rewards: list[float], path: Path = REWARD_PLOT_PATH) -> None:
    """Create and save a reward-vs-episode graph."""
    if not rewards:
        print("No episode rewards were recorded, so no plot was created.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Performance: Reward vs Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_training_summary(
    device: str,
    total_timesteps: int,
    elapsed_seconds: float,
    rewards: list[float],
    path: Path = SUMMARY_PATH,
) -> None:
    """Save a simple text summary for the graph/report team."""
    average_reward = sum(rewards) / len(rewards) if rewards else 0.0
    best_reward = max(rewards) if rewards else 0.0

    with path.open("w", encoding="utf-8") as file:
        file.write("Highway-Env PPO Training Summary\n")
        file.write("================================\n")
        file.write(f"Device: {device}\n")
        file.write(f"Total timesteps: {total_timesteps}\n")
        file.write(f"Elapsed seconds: {elapsed_seconds:.2f}\n")
        file.write(f"Approx FPS: {total_timesteps / max(elapsed_seconds, 1):.2f}\n")
        file.write(f"Episodes recorded: {len(rewards)}\n")
        file.write(f"Average episode reward: {average_reward:.2f}\n")
        file.write(f"Best episode reward: {best_reward:.2f}\n")


def now() -> float:
    """Return current time for simple runtime tracking."""
    return time.perf_counter()
