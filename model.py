import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from config import PPO_HYPERPARAMETERS, SEED


def get_training_device() -> str:
    """Return cuda when available, otherwise cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_ppo_model(env) -> BaseAlgorithm:
    """Create a PPO model for the configured Highway-Env environment."""
    device = get_training_device()
    print(f"Training device selected: {device}")
    if device == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will use CPU.")

    return PPO(
        env=env,
        seed=SEED,
        device=device,
        **PPO_HYPERPARAMETERS,
    )
