from pathlib import Path

ENV_NAME: str = "highway-v0"

BASE_DIR: Path = Path(__file__).resolve().parent
MODEL_DIR: Path = BASE_DIR / "models"
ASSET_DIR: Path = BASE_DIR / "assets"
LOG_DIR: Path = BASE_DIR / "logs"

# Fast settings for coursework/testing. Increase later if you want a stronger agent.
TOTAL_TIMESTEPS: int = 20_000
HALF_TIMESTEPS: int = TOTAL_TIMESTEPS // 2
N_ENVS: int = 4
EVAL_EPISODES: int = 5
SEED: int = 42

PPO_HYPERPARAMETERS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "verbose": 1,
}

MODEL_PATHS = {
    "untrained": MODEL_DIR / "ppo_highway_untrained.zip",
    "half_trained": MODEL_DIR / "ppo_highway_half_trained.zip",
    "fully_trained": MODEL_DIR / "ppo_highway_fully_trained.zip",
}

REWARD_PLOT_PATH: Path = ASSET_DIR / "reward_plot.png"
REWARD_LOG_PATH: Path = LOG_DIR / "episode_rewards.csv"
SUMMARY_PATH: Path = LOG_DIR / "training_summary.txt"
