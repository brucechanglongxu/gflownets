# GFlowNets

GFlowNets are generative models that learn to sample compositional objects (e.g. molecules, graphs, sequences) _proportionally to a reward_ using a learned flow through a state space. They differ from traditional generative models by using a _trajectory-based_ approach inspired by reinforcement learning (but not optimizing expected return). GFlowNets projects require the following:

- Environment/state space definition
- Trajectory generation model
- Flow learning objective (e.g. trajectory balance, detailed balance)
- Training loop with sampling and updates
- Reward function
- Evaluation suite 

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
