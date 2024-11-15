# Naive-Framework
A naive framework about Generalized RL

## Introduction

In the research of Reinforcement Learning (RL), enhancing the generalization ability of agents is an urgent issue to be addressed. Especially in multi-environment scenarios, models trained in a single environment struggle to cope with highly variable situations. To this end, we propose an innovative Adaptive Cross-Environment Reinforcement Training (ARC) framework. This framework allows agents to alternate or mix learning in multiple similar yet different environments through dynamic strategy switching. This design enhances the agents' generalization ability across different environments and provides a flexible and efficient training solution in multi-environment scenarios. The framework supports multiple reinforcement learning algorithms and different types of environments, offering high scalability.

## Installation

```bash
TBD
```

## Quick Start

### Example Configuration

```toml
n_timestep = 10_000_000                            # total training timesteps needed
c_lr = 0.01                                        # learning rate
cap = 10000                                        # cap for environment switching
c_transition_loss = 0.5                            # transition loss coefficient
algorithm = "rainbow"                              # algorithm identifier
policy = "CnnPolicy"                               # policy identifier
eval_freq = 10                                     # evaluation frequency
eval_episodes = 5                                  # number of evaluation episodes
seed = 42                                          # random seed
device = "cuda"                                    # computation device
env_weights = [0.5, 0.5]                           # environment weights
env_ids = ["ALE/DemonAttack-v5", "ALE/Phoenix-v5"] # environments to train
switching_algorithm = "algo2"                      # switching algorithm
```
### Example Command
```bash
# Run the experiment with the configuration file demo.toml and DI-engine as the reinforcement learning framework
python3 exp.py --config=demo.toml --rlf=ding
# Evaluate the trained model
python3 eval.py --config=demo.toml --rlf=ding
```

## c[a]t
![image](./cat.png)