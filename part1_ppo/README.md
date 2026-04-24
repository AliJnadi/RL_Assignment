# Part 1 – PPO & SAC Training on FetchPickAndPlace

## Overview

This folder contains the training and evaluation code for the FetchPickAndPlace task (MuJoCo).  
We provide two algorithms:

- **PPO** (on‑policy) with sparse and dense reward wrappers.
- **SAC** (off‑policy) with dense reward.

All models are saved together with their observation normalisers (`VecNormalize`).  
Training logs are stored in TensorBoard format.
