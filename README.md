# Adversarial Attacks on Deep Algorithmic Trading Policies

This repository provides the implementation of the research paper "[Adversarial Attacks on Deep Algorithmic Trading Policies](https://arxiv.org/pdf/2010.11388v1)" by Yaser Faghan, Nancirose Piazza, Vahid Behzadan, and Ali Fathi. The paper explores the vulnerability of Deep Reinforcement Learning (DRL) agents in algorithmic trading systems to adversarial attacks. The authors propose a threat model and two novel attack techniques that manipulate the performance of DRL-based trading policies at test time. This repository reproduces the key ideas and results from the paper using Python and PyTorch.

---

## Core Concept

Algorithmic trading systems, particularly those powered by Deep Reinforcement Learning (DRL), have gained significant traction in financial markets. These systems autonomously make trading decisions to maximize profits. However, DRL models are known to be vulnerable to adversarial perturbations, where small, carefully crafted changes to the input can lead to suboptimal or manipulated decisions.

The paper introduces a **threat model** tailored to trading agents and proposes two **adversarial attack techniques**:
1. **Reward Manipulation Attacks**: These perturb the reward signals received by the DRL agent during trading, leading to incorrect learning or decision-making.
2. **State Perturbation Attacks**: These directly modify the input state (e.g., price series) fed into the agent to influence its trading policy.

The authors demonstrate the effectiveness of these attacks against both benchmark Deep Q-Network (DQN) agents and real-world trading environments. The result highlights the potential risks of deploying DRL-based trading systems in adversarial settings.

---

## Repository Overview

This repository contains a Python implementation using PyTorch to demonstrate the adversarial attacks described in the paper. The code is organized as follows:

### 1. **Agent Implementation**
The repository includes a baseline **Deep Q-Network (DQN)** trading agent. This agent is trained to make buy/sell/hold decisions based on historical price data to maximize profit.

- `dqn_agent.py`: Defines the architecture and behavior of the DQN agent.
- `training_pipeline.py`: Contains the training logic for the DQN agent, including reward computation and environment interaction.

### 2. **Adversarial Attack Techniques**
Two attack methods are implemented to manipulate the DQN agent:
- **Reward Manipulation Attack**:
  - Alters the reward signals received by the agent to mislead its learning process or trading behavior.
  - Implemented in `reward_attack.py`.
- **State Perturbation Attack**:
  - Introduces carefully crafted noise into the input state (e.g., price data) to manipulate the agent's policy.
  - Implemented in `state_attack.py`.

### 3. **Simulated Trading Environment**
A custom trading environment is built to simulate the interaction between the agent and the market.
- `trading_env.py`: Implements the trading environment, including market simulation, state generation, and reward calculation.

### 4. **Evaluation and Visualization**
Scripts to evaluate the performance of the DQN agents under normal conditions and adversarial attacks:
- `evaluate_agent.py`: Runs the trained agent in the trading environment and reports performance metrics.
- `visualization.py`: Generates plots to compare the agent's performance with and without adversarial attacks.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repo-name.git
   cd repo-name
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the DQN Agent
To train the DQN agent on a simulated trading environment:
```bash
python training_pipeline.py
```

### Running Adversarial Attacks
To evaluate the impact of adversarial attacks on the trained agent:
- **Reward Manipulation Attack**:
  ```bash
  python reward_attack.py
  ```
- **State Perturbation Attack**:
  ```bash
  python state_attack.py
  ```

### Visualizing Results
To visualize the agent's performance under different scenarios:
```bash
python visualization.py
```

---

## Results

The paper demonstrates that the proposed adversarial attacks can significantly degrade the performance of DRL-based trading agents. Key findings include:
- Both reward manipulation and state perturbation attacks are highly effective in altering the agent's trading decisions.
- These attacks pose a critical risk to the deployment of DRL systems in adversarial financial environments.

Detailed performance metrics and visualizations are provided in the paper and can be reproduced using the scripts in this repository.

---

## Future Work

The following extensions are suggested for future research:
- Exploring defense mechanisms to make DRL trading agents robust against adversarial attacks.
- Extending the threat model to other types of DRL agents (e.g., PPO, A3C) and trading strategies.
- Evaluating the impact of adversarial attacks in real-world financial markets with higher complexity.

---

## Citation

If you find this work useful, please cite the original paper:
```
@article{faghan2020adversarial,
  title={Adversarial Attacks on Deep Algorithmic Trading Policies},
  author={Faghan, Yaser and Piazza, Nancirose and Behzadan, Vahid and Fathi, Ali},
  journal={arXiv preprint arXiv:2010.11388},
  year={2020}
}
```

---

## License

This repository is released under the [MIT License](LICENSE). Please refer to the license file for more details.