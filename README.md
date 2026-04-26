# Robotic Manipulation with PPO / SAC and ROS2 Fallback

## Project Overview

This repository contains the implementation and evaluation of two related tasks:

1. **[Part 1](https://github.com/AliJnadi/RL_Assignment/tree/main/part1_ppo) – Training PPO & SAC on FetchPickAndPlace**  
   Train reinforcement learning agents (PPO with sparse/dense rewards, SAC with dense reward) on the [`FetchPickAndPlace-v4`](https://robotics.farama.org/envs/fetch/pick_and_place/) environment (MuJoCo).

2. **[Part 2](https://github.com/AliJnadi/RL_Assignment/tree/main/part2_ros) – ROS2 Fallback**  
   A ROS2‑based closed‑loop control system that drives the same simulator.  
   Implements step‑numbered observation/action matching to avoid stale actions.

## Installation

Clone the repository:
```bash
git clone https://github.com/AliJnadi/RL_Assignment.git
```
