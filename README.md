# Goal-Conditioned Reinforcement Learning with LTL Objectives

This repository implements a pipeline for training and executing goal-conditioned reinforcement learning agents that can follow Linear Temporal Logic (LTL) specifications. The pipeline consists of four main stages:

1. Training goal-conditioned PPO agents on fixed sequences
2. Verifying sequence performance 
3. Training a goal-conditioned value function (GCVF)
4. Executing LTL specifications


## 1. Training Goal-Conditioned PPO Agents

First, we train PPO agents to handle fixed goal sequences using `sequence.py`. The script trains separate models for each possible sequence of three goals (red, green, yellow).

```bash
python sequence.py \
    --unity_env_path /path/to/unity/env \
    --timesteps 2000000 \
    --num_envs 4 \
    --device cuda

# The script will:
# - Train models for all 6 possible sequences
# - Save models to models/sequence_{0-5}_final
# - Create checkpoints in models/sequence_{idx}/stage_{stage_idx}
```

## 2. Verifying Sequence Performance

After training, verify the performance of trained models using `verify_sequences.py`:

```bash
# Verify all sequences
python verify_sequences.py \
    --model_path models/sequence_0_final \
    --env_path /path/to/unity/env \
    --n_episodes 20 \
    --max_steps 1000

# Or verify specific sequences
python verify_sequences.py \
    --model_path models/sequence_0_final \
    --env_path /path/to/unity/env \
    --sequences "red,green,yellow" "yellow,red,green"
```

This will generate:
- Detailed analysis of success rates
- Goal achievement statistics
- Visualization plots saved as 'sequence_analysis.png'

## 3. Training GCVF

After collecting trajectories during PPO training, train the goal-conditioned value function:

```bash
python train_gcvf.py \
    --dataset_path datasets/traj_dataset.pt \
    --device cuda \
    --batch_size 512 \
    --lr 0.0004 \
    --num_epochs 1000
```

The trained GCVF model will be saved to `models/goal-conditioned/gcvf.pth`.

## 4. Executing LTL Specifications

Finally, use the trained models to execute LTL specifications:

```bash
python main.py \
    --ltl "!y U (j && (!w U r))" \
    --unity_env_path /path/to/unity/env \
    --gcvf_path models/goal-conditioned/gcvf.pth \
    --render \
    --device cuda
```

The execution process:
1. Converts LTL to Büchi automaton using GLTL2BA
2. Uses GCVF to find optimal goal sequence
3. Selects appropriate trained sequence model
4. Executes the sequence in Unity environment

## Example LTL Formulas

Some example LTL formulas to try:
- `!y U (j && (!w U r))`: Reach j (Jetblack) while avoiding y (Yellow), then reach r (Red) while avoiding w (White)
- `F(j && F(w && F(r)))`: Reach j, then w, then r in sequence
- `G(!w)`: Always avoid w
- `GF(r)`: Repeatedly reach r infinitely often

## Folder Structure

```
.
├── models/
│   ├── sequence_0_final/        # Trained PPO models for each sequence
│   ├── sequence_1_final/
│   └── goal-conditioned/
│       └── gcvf.pth            # Trained GCVF model
├── datasets/
│   └── traj_dataset.pt         # Collected trajectories
├── sequence.py                  # PPO training script
├── verify_sequences.py         # Sequence verification script
├── train_gcvf.py              # GCVF training script
└── main.py                    # LTL execution script
```

## Notes

- Ensure Unity environment is running before executing scripts
- Default training uses 4 parallel environments
- Training one sequence takes approximately 2M timesteps
- GCVF training typically requires 1000 epochs for convergence
- When rendering, environment runs at real-time speed; disable rendering for faster execution

## Troubleshooting

- If Unity process doesn't close properly, use `--worker_id` with different values
- For CUDA out of memory errors, reduce batch size or number of environments
- Check logs/ directory for training progress and debugging information
