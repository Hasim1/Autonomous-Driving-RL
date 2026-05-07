import gymnasium as gym
import highway_env


# Name of the Highway-Env environment
ENV_NAME = "highway-v0"


# Environment configuration
# This defines the observation/state space, action space, and reward settings.
ENV_CONFIG = {
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

    # Highway settings
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,

    # Reward function components
    # The agent is rewarded for speed and punished for unsafe behavior.
    "collision_reward": -5.0,
    "high_speed_reward": 1.0,
    "right_lane_reward": 0.1,
    "lane_change_reward": -0.2,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
}


# States observed by the agent
STATES_DESCRIPTION = {
    "presence": "Indicates whether a vehicle exists in the observation.",
    "x": "Relative longitudinal position of the vehicle.",
    "y": "Relative lateral or lane position of the vehicle.",
    "vx": "Relative longitudinal speed of the vehicle.",
    "vy": "Relative lateral speed of the vehicle.",
}


# Actions available to the agent
ACTIONS_DESCRIPTION = {
    0: "LANE_LEFT: Move to the left lane.",
    1: "IDLE: Keep the same lane and behavior.",
    2: "LANE_RIGHT: Move to the right lane.",
    3: "FASTER: Increase speed.",
    4: "SLOWER: Decrease speed.",
}


def create_environment(render_mode=None):
    """
    Create and configure the highway-v0 environment.

    The environment represents a multi-lane highway where the ego vehicle
    learns to drive safely, maintain good speed, and avoid unsafe behavior.
    """

    env = gym.make(ENV_NAME, render_mode=render_mode)
    env.unwrapped.configure(ENV_CONFIG)
    env.reset()

    return env


def calculate_initial_reward(
    speed,
    collision=False,
    off_road=False,
    unnecessary_lane_change=False
):
    """
    Initial reward function for the autonomous driving agent.

    Reward formula:
        R = alpha * speed
            - beta * collision
            - gamma * off_road
            - delta * unnecessary_lane_change

    The reward function encourages the agent to drive at a good speed,
    while penalizing crashes, going off-road, and unnecessary lane changes.
    """

    alpha = 1.0    # speed reward weight
    beta = 5.0     # collision penalty weight
    gamma = 3.0    # off-road penalty weight
    delta = 0.2    # unnecessary lane-change penalty weight

    reward = alpha * speed

    if collision:
        reward -= beta

    if off_road:
        reward -= gamma

    if unnecessary_lane_change:
        reward -= delta

    return reward


def print_environment_info():
    """
    Print states, actions, and reward function information.
    """

    print("Environment:", ENV_NAME)

    print("\nStates:")
    for state, description in STATES_DESCRIPTION.items():
        print(f"- {state}: {description}")

    print("\nActions:")
    for action_id, description in ACTIONS_DESCRIPTION.items():
        print(f"- {action_id}: {description}")

    print("\nReward Function:")
    print("R = alpha * speed - beta * collision - gamma * off_road - delta * lane_change")
    print("alpha = 1.0")
    print("beta = 5.0")
    print("gamma = 3.0")
    print("delta = 0.2")


if __name__ == "__main__":
    env = create_environment()

    print("Environment created successfully.")
    print_environment_info()

    observation, info = env.reset()

    print("\nInitial observation shape:")
    print(observation.shape)

    sample_reward = calculate_initial_reward(
        speed=25,
        collision=False,
        off_road=False,
        unnecessary_lane_change=False
    )

    print("\nSample reward for safe driving at speed 25:")
    print(sample_reward)

    env.close()