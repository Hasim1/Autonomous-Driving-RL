from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config import EVAL_EPISODES, MODEL_PATHS
from environment import create_environment


def evaluate_model(model_path: Path, episodes: int = EVAL_EPISODES) -> float:
    """Run evaluation episodes and return the average reward."""
    env = create_environment(render_mode=None)
    model = PPO.load(model_path, env=env)
    rewards: list[float] = []

    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards))


def evaluate_all_models() -> None:
    """Evaluate untrained, half-trained, and fully-trained checkpoints."""
    for stage, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"Missing model for {stage}: {path}")
            continue
        average_reward = evaluate_model(path)
        print(f"{stage}: average reward = {average_reward:.2f}")


if __name__ == "__main__":
    evaluate_all_models()
