# Autonomous Driving using Reinforcement Learning 

This repository contains the final project for training an autonomous driving agent using Reinforcement Learning.
The agent learns to navigate a highway safely and efficiently.

##  Team Members & Roles
* **•	Noor aldeen abuzannad  - 2105678 **: Environment Setup & Reward Function Design
* **•	Adnan Azizia           - 2017206 **: Model Training (PPO)
* **•	Fadhl al-fadhili       - 2363825  **: Evaluation, Video Generation & Graphs
* **•	Hashim ALtabatabee     - 2105338 **: GitHub Repository Management & Documentation

---

##  Project Goal
The main objective is to train a self-driving car to achieve a balance between:
1. **Safety:** Avoiding collisions.
2. **Speed:** Driving at an optimal speed.
3. **Lane Discipline:** Staying in the appropriate lane.

---

## The Reward Function
To teach the agent how to drive, we designed a custom reward function.

$$
R_t = \alpha \cdot v_t - \beta \cdot C_t - \gamma \cdot O_t - \delta \cdot L_t
$$

* $v_t$: Speed of the vehicle.
* $C_t$: Collision penalty.
* $O_t$: Off-road penalty.
* $L_t$: Unnecessary lane change penalty.

---

##  Results & Evaluation

### Training Performance (Reward Graph)
Here is the graph showing how the agent's reward increased over time:

![Reward Graph](assets/reward_plot.png)

### Agent Behavior (Untrained vs. Fully Trained)
### Agent Behavior (Untrained vs. Fully Trained)

**1. Untrained Agent (Before Training)**
The car moves randomly, makes poor decisions, and frequently crashes.

<video src="assets/untrained0.mp4" controls="controls" width="100%"></video>

**2. Fully Trained Agent (After Training)**
The car drives smoothly, changes lanes smartly to avoid traffic, and maintains a high optimal speed!

<video src="assets/fully_trained0.mp4" controls="controls" width="100%"></video>

---

## How to Run Locally.
1. Clone the repository:
   ```bash
   git clone [https://github.com/Hasim1/Autonomous-Driving-RL.git](https://github.com/Hasim1/Autonomous-Driving-RL.git)