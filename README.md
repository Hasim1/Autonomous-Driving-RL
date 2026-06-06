# Autonomous Driving using Reinforcement Learning 

This repository contains the final project for training an autonomous driving agent using Reinforcement Learning. The agent learns to navigate a highway safely and efficiently.

**Course Code:** CMP4501  
**Selected Project Track:** Option A - Autonomous Driving with Highway-Env  

## Team Members & Roles
* **Noor aldeen abuzannad - 2105678:** Environment Setup & Reward Function Design
* **Adnan Azizia - 2017206:** Model Training (PPO)
* **Fadhl al-fadhili - 2363825:** Evaluation, Video Generation & Graphs
* **Hashim ALtabatabee - 2105338:** GitHub Repository Management & Documentation

---

## 1. Visual Proof (Agent Evolution)

**A. Untrained Agent (Before Training)**
The car moves randomly, makes poor decisions, and frequently crashes.
<video src="assets/untrained0.mp4" controls="controls" width="100%"></video>

**B. Half-Trained Agent (Mid-Training)**
The agent shows partial learning. It attempts to drive and avoid collisions but still makes noticeable mistakes.
<video src="assets/half_trained.mp4" controls="controls" width="100%"></video>

**C. Fully Trained Agent (After Training)**
The car drives smoothly, changes lanes smartly to avoid traffic, and maintains a high optimal speed.
<video src="assets/fully_trained0.mp4" controls="controls" width="100%"></video>

---

## 2. Methodology

### a. The Reward Function
To teach the agent how to drive safely and efficiently, we designed the following custom reward function:

$$R_t = \alpha \cdot v_t - \beta \cdot C_t - \gamma \cdot O_t - \delta \cdot L_t$$

*   $v_t$: Speed of the vehicle.
*   $C_t$: Collision penalty.
*   $O_t$: Off-road penalty.
*   $L_t$: Unnecessary lane change penalty.

**Justification:** This reward function is highly suitable for a multi-objective autonomous driving problem. By assigning a positive weight ($\alpha$) to speed, the agent is encouraged to reach its destination quickly. However, to ensure safety, the collision penalty ($\beta$) is set significantly higher to strictly penalize crashes. The lane change penalty ($\delta$) prevents the agent from driving erratically and zig-zagging between lanes, promoting a smooth driving experience.

### b. The Model
We selected the **Proximal Policy Optimization (PPO)** algorithm for this project. PPO is an excellent choice for continuous state spaces and discrete action environments like highway driving because it strikes a perfect balance between sample efficiency and ease of tuning, avoiding the catastrophic performance drops common in older algorithms.

**Key Hyperparameters & Architecture:**
*   **Policy Architecture:** `MlpPolicy` (Multi-Layer Perceptron). Since our environment relies on kinematics (numerical data) rather than raw pixels, a standard neural network is highly efficient.
*   **Learning Rate:** $3 \times 10^{-4}$, providing stable and gradual convergence.
*   **Batch Size & n_steps:** Set to 256, ensuring enough experiences are collected before updating the network.
*   **Discount Factor (Gamma):** $0.99$, encouraging the agent to prioritize long-term survival over immediate, short-term rewards.

### c. States and Actions
**Observation Space (States):**
The agent observes the environment through a "Kinematics" feature vector representing the 5 closest vehicles. For each vehicle, the agent receives:
*   `presence`: A boolean indicating if the vehicle exists.
*   `x` & `y`: Relative longitudinal and lateral positions.
*   `vx` & `vy`: Relative longitudinal and lateral speeds.

**Action Space:**
The agent interacts with the environment using a `DiscreteMetaAction` space, consisting of 5 high-level decisions:
*   `0`: Change lane to the left.
*   `1`: Idle (maintain current speed and lane).
*   `2`: Change lane to the right.
*   `3`: Accelerate (go faster).
*   `4`: Decelerate (go slower).

---

## 3. Training Analysis

### a. Reward Graph
Here is the graph showing how the agent's reward increased over time:

![Reward Graph](assets/reward_plot.png)

### b. Commentary
Based on the `reward_plot.png` generated during our training phase, the learning dynamics of the PPO agent are clearly visible:
*   **Early Training (Episodes 0 - 400):** The training was highly unstable, with rewards frequently dropping very low. This represents the *exploration phase*, where the untrained agent took random actions, leading to frequent collisions and poor lane choices.
*   **Convergence (Episodes 400 - 800):** Performance began to improve significantly. The agent learned the correlation between high speed and avoiding collisions, stabilizing the average reward at a higher level.
*   **Late Training & Plateaus (Episodes 800+):** The learning curve plateaued in a highly stable manner. The occasional sharp drops (spikes downwards) are completely normal in PPO; they represent the agent occasionally exploring sub-optimal actions or encountering unavoidable traffic congestion.

---

## 4. Challenges and Failures

**Challenge 1: CPU Thread Oversubscription**
*   **Difficulty:** Initially, the training process was extremely slow. Because the `MlpPolicy` neural network is relatively small, PyTorch was attempting to parallelize the operations across too many CPU cores, leading to "thread oversubscription" which choked the CPU and reduced performance.
*   **Solution:** We solved this by explicitly limiting PyTorch's multithreading capabilities in `train.py` using `os.environ["OMP_NUM_THREADS"] = "1"`. This drastically improved our frames-per-second (FPS) and made training highly efficient.

**Challenge 2: Unintended Reward Exploitation**
*   **Difficulty:** Early on, the agent learned an unintended behavior: it would drive at maximum speed but constantly switch lanes back and forth to avoid slowing down, resulting in reckless driving.
*   **Solution:** We addressed this by fine-tuning the reward function to include a specific penalty for lane changes (`lane_change_reward = -0.2`). This forced the agent to value lane discipline, resulting in much smoother behavior.

---

## How to Run Locally

1. Clone the repository: `git clone https://github.com/Hasim1/Autonomous-Driving-RL.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the evaluation to see the fully trained agent: `python evaluate.py`