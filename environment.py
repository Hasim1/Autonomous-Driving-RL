from typing import Any

import gymnasium as gym
import highway_env

from config import ENV_NAME


ENV_CONFIG: dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 8,
    "duration": 20,
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "collision_reward": -5.0,
    "high_speed_reward": 1.0,
    "right_lane_reward": 0.1,
    "lane_change_reward": -0.2,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
}


STATES_DESCRIPTION: dict[str, str] = {
    "presence": "Indicates whether a nearby vehicle exists in the observation.",
    "x": "Relative longitudinal position of the vehicle.",
    "y": "Relative lateral or lane position of the vehicle.",
    "vx": "Relative longitudinal speed of the vehicle.",
    "vy": "Relative lateral speed of the vehicle.",
}


ACTIONS_DESCRIPTION: dict[int, str] = {
    0: "LANE_LEFT: Move to the left lane.",
    1: "IDLE: Keep the current lane and speed.",
    2: "LANE_RIGHT: Move to the right lane.",
    3: "FASTER: Increase speed.",
    4: "SLOWER: Decrease speed.",
}


def create_environment(render_mode: str | None = None) -> gym.Env:
    """Create and configure the Highway-Env environment.

    render_mode must stay None during training for speed.
    Use render_mode="rgb_array" only when recording videos.
    """
    env = gym.make(ENV_NAME, render_mode=render_mode)
    env.unwrapped.configure(ENV_CONFIG)
    env.reset()
    return env


def calculate_initial_reward(
    speed: float,
    collision: bool = False,
    off_road: bool = False,
    unnecessary_lane_change: bool = False,
) -> float:
    """Example reward equation used for the methodology explanation."""
    alpha = 1.0
    beta = 5.0
    gamma = 3.0
    delta = 0.2

    reward = alpha * speed
    if collision:
        reward -= beta
    if off_road:
        reward -= gamma
    if unnecessary_lane_change:
        reward -= delta
    return reward


def print_environment_info() -> None:
    """Print states, actions, and reward-function information."""
    print("Environment:", ENV_NAME)

    print("\nStates:")
    for state, description in STATES_DESCRIPTION.items():
        print(f"- {state}: {description}")

    print("\nActions:")
    for action_id, description in ACTIONS_DESCRIPTION.items():
        print(f"- {action_id}: {description}")

    print("\nReward Function:")
    print("R = alpha * speed - beta * collision - gamma * off_road - delta * lane_change")
    print("alpha = 1.0, beta = 5.0, gamma = 3.0, delta = 0.2")


if __name__ == "__main__":
    test_env = create_environment()
    print("Environment created successfully.")
    print_environment_info()
    observation, _ = test_env.reset()
    print("\nInitial observation shape:", observation.shape)
    print("Sample reward:", calculate_initial_reward(speed=25))
    test_env.close()
