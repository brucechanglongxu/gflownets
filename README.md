# GFlowNets

GFlowNets are generative models that learn to sample compositional objects (e.g. molecules, graphs, sequences) _proportionally to a reward_ using a learned flow through a state space. They differ from traditional generative models by using a _trajectory-based_ approach inspired by reinforcement learning (but not optimizing expected return). GFlowNets projects require the following:

- **Environment/state space definition:** This defines how the state is constructed and which actions are legal at each step. `MoleculeEnv` add atoms/bonds incrementally, `GraphEnv` build graphs node by node, `SequenceEnv` build sequence/token chains. We must implement `reset()`, `step(action)`, `get_valid_actions()`, `is_terminal()`, `get_state_representation()`.
- **Trajectory generation model/Flow learning objective (e.g. trajectory balance, detailed balance):** `models/` contains the policy and flow networks (forward policy $$P(s_{t+1}|s_t)$$ the backward policy $$P(s_t|s_{t+1})$$ and the flow function $$F(s)$$ in trajectory balance. This is usually modeled as MLPs (for tabular/sequences), GNNs (for molecules/graphs) or transformers (for longer dependencies). 
- **Training loop with sampling and updates:** `training/` contains the core learning objectives, where each trainer implements a different learning algorithm. _Trajectory Balance (TB)_ trains a single forward policy/flow. _Detailed Balance (DB)_ trains both forward and backward policies. _SubTB/DB_ is used for parital trajectories. All loss functions, optimizers and logging also lives in this subdirectory. 
- **Reward function:** For molecules we would use QED, logP, binding affinity (oracle call). For sequences we would use BLEU score, perplexity, and for graphs we would use connectivity or spectral properties. 
- **Evaluation suite:** `utils/` contains the sampler, replay buffer, logger and metrics. `scripts/` contains entrpy points `train.py` loads config, environment, model and trainer, `evaluate.py` generates samples and evaluates reward, `visualize.py` embeds trajectories or objects. 

Our directory structure is as follows:
```
gflownet/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── configs/              # Training and model configs (YAML/JSON)
│   └── molecule.yaml     # Example: reward function, model, env
│
├── gflownet/             # Core library
│   ├── envs/             # Environments (state/action definitions)
│   │   ├── base_env.py
│   │   ├── molecule_env.py
│   │   └── ...
│   │
│   ├── models/           # Policy/backward/forward models
│   │   ├── base_model.py
│   │   ├── mlp_policy.py
│   │   └── gnn_policy.py
│   │
│   ├── training/         # Training objectives and loops
│   │   ├── base_trainer.py
│   │   ├── tb_trainer.py     # Trajectory Balance
│   │   ├── db_trainer.py     # Detailed Balance
│   │   └── replay_buffer.py
│   │
│   ├── rewards/          # Reward function definitions
│   │   ├── base_reward.py
│   │   └── qed_reward.py
│   │
│   ├── utils/            # Sampling, logging, checkpointing
│   │   ├── logger.py
│   │   ├── sampler.py
│   │   ├── metrics.py
│   │   └── ...
│
├── scripts/              # Training scripts, evaluation, etc.
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
├── notebooks/            # Jupyter notebooks for demos and dev
│   ├── demo_molecule.ipynb
│   └── ablation_experiments.ipynb
│
├── tests/                # Unit tests
│   ├── test_envs.py
│   ├── test_trainers.py
│   └── ...
│
└── examples/             # Pretrained models or result visualizations
    └── molecule_gen/
```
