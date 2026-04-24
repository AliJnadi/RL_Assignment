# PPO & SAC Training on FetchPickAndPlace

## Overview

This folder contains the [training](https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/train.py) and [evaluation](https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/eval.py) code for the [FetchPickAndPlace](https://robotics.farama.org/envs/fetch/pick_and_place/) task (MuJoCo).  

The `FetchPickAndPlace-v4` environment simulates a 7‑DoF Fetch robot in MuJoCo.  
**Task**: The robot must grasp a small cube from a table, lift it to a target height (0.5 m above the table), the episode never terminated only trancated after 50 steps.  
**Observation**: 25‑dimensional vector (end‑effector pose, object pose, gripper state, relative positions, etc.).  
**Action**: 4‑dimensional continuous (x, y, z displacement + gripper open/close).  
**Reward**: Sparse (binary success + small proximity) or dense (shaped distance‑to‑goal).

We provide two algorithms:

- **PPO** (on‑policy) with sparse and dense reward wrappers.
- **SAC** (off‑policy) with dense reward.

All [models](https://github.com/AliJnadi/RL_Assignment/tree/main/part1_ppo/trained_models) are saved together with their observation normalisers (`VecNormalize`).  
[Training logs](https://github.com/AliJnadi/RL_Assignment/tree/main/part1_ppo/tensorboard) are stored in TensorBoard format.

## Environment & Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training
All training commands use [train.py](https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/train.py). The script accepts the following arguments:

Argument	Description	Default
--algo	PPO or SAC	PPO
--mode	sparse or dense	sparse
--seeds	list of random seeds	0 1 2
--timesteps	total timesteps per seed	6000000 (PPO), 800000 (SAC)
--n_envs	number of parallel envs	2 (PPO), 1 (SAC)

1. PPO with sparse reward (6M steps, 2 envs)

```bash
python train.py --algo PPO --mode sparse --seeds 0 1 2 --timesteps 6000000 --n_envs 2
```

2. PPO with dense reward

```bash
python train.py --algo PPO --mode dense --seeds 0 1 2 --timesteps 6000000 --n_envs 2
```

3. SAC with dense reward (800k steps, 1 env)

```bash
python train.py --algo SAC --mode dense --seeds 0 1 2 --timesteps 800000 --n_envs 1
```

All hyperparameters (learning rates, network architectures, etc.) are hard‑coded inside train.py.

## TensorBoard Logs
While training, logs are saved under `tensorboard/algo/mode/seed_X/`.
View them with:

```bash
tensorboard --logdir tensorboard/
```

## Evaluation
After training, evaluate a model and generate GIFs using [eval.py](https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/eval.py):

```bash
# Evaluate all seeds of PPO sparse
python eval.py --algo PPO --mode sparse --folder_name PPO_models

# Evaluate a specific seed of SAC dense
python eval.py --algo SAC --mode dense --folder_name SAC_models --specific_seed 0
```

The script will:
* Run 100 deterministic episodes and 100 non‑deterministic episodes.
* Print success rates.
* Save a [GIF](https://github.com/AliJnadi/RL_Assignment/tree/main/part1_ppo/gifs) (5 episodes) for each seed in gifs/algo/

## Seeds & Randomness
- Training seeds: explicitly set via --seeds. We used 0,1,2.
- Evaluation: no global seed is fixed, allowing the environment’s natural stochasticity (object positions, goal variations) to be part of the performance measurement.

## Saved Policy Checkpoints
Final models are saved in:
* PPO_models/sparse/PickPlace_seed_<seed>.zip
* PPO_models/dense/PickPlace_seed_<seed>.zip
* SAC_models/dense/PickPlace_seed_<seed>.zip

Each model is accompanied by a vecnorm_seed_<seed>.pkl file containing running statistics for observation normalisation.

## Demonstrations (GIFs)
Evaluation produces GIFs for each seed. The following show performance of all models:

## Environment: FetchPickAndPlace (MuJoCo)

The `FetchPickAndPlace-v4` environment simulates a 7‑DoF Fetch robot in MuJoCo.  
**Task**: The robot must grasp a small cube from a table, lift it to a target height (0.5 m above the table), and release it.  
**Observation**: 25‑dimensional vector (end‑effector pose, object pose, gripper state, relative positions, etc.).  
**Action**: 4‑dimensional continuous (x, y, z displacement + gripper open/close).  
**Reward**: Sparse (binary success + small proximity) or dense (shaped distance‑to‑goal).

We trained three policies:

- **PPO with sparse reward**
- **PPO with dense reward**
- **SAC with dense reward**

Below are GIFs of the best final policies (seed 0 for PPO, seed 2 for SAC) performing the task successfully.

---

### PPO sparse

<div align="center">
  <img src="https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/gifs/PPO/sparse/seed_0.gif" width="500"/>
  <br/>
  <sub>Fetch robot picking and placing the cube (sparse reward)</sub>
</div>

### PPO dense

<div align="center">
  <img src="https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/gifs/PPO/dense/seed_0.gif" width="500"/>
  <br/>
  <sub>Smoother behaviour due to dense reward shaping</sub>
</div>

### SAC dense

<div align="center">
  <img src="https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/gifs/SAC/dense/seed_2.gif" width="500"/>
  <br/>
  <sub>Off‑policy SAC with dense reward achieves highest success rate</sub>
</div>

### PPO sparse
![PPO sparse](https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/gifs/PPO/sparse/seed_0.gif)

### PPO dense
![PPO dense](https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/gifs/PPO/dense/seed_0.gif)

### SAC dense
![SAC dense](https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/gifs/SAC/dense/seed_2.gif)

## Notes on Reproducibility
* All random number generators are seeded at the start of each training run (via Stable‑Baselines3 seed argument and environment reset(seed=...)).

* The evaluation script does not call any set_seed() function – this ensures the reported success rates reflect real‑world performance across varied initial conditions.

* The exact environment wrapper (RewardWrapperSparse / RewardWrapperDense) is provided in [RewardWrapper.py](https://github.com/AliJnadi/RL_Assignment/blob/main/part1_ppo/RewardWrapper.py).
