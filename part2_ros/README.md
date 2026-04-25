# ROS2 Fallback: Closed‑Loop Control of FetchPickAndPlace

## Overview

This ROS2 package implements a closed‑loop control system that drives the `FetchPickAndPlace` simulator (MuJoCo) using a pre‑trained policy SAC.  
It consists of two nodes:

- [**`env_node`**](https://github.com/AliJnadi/RL_Assignment/blob/main/part2_ros/src/rl_control/rl_control/env_node.py) – owns the environment, publishes observations, subscribes to actions, and steps the simulation.
- [**`agent_node`**](https://github.com/AliJnadi/RL_Assignment/blob/main/part2_ros/src/rl_control/rl_control/agent_node.py) – loads a trained policy, subscribes to observations, runs inference, and publishes actions.

## Build & Run Instructions

### Prerequisites

- ROS2 [Jazzy](https://docs.ros.org/en/jazzy/index.html).
- [Gymnasium](https://gymnasium.farama.org/index.html).
- [Gymnasium robotics](https://robotics.farama.org/index.html).
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/).

### Build the ROS2 Workspace

```bash
cd part2_ros
colcon build --packages-select rl_control
source install/setup.bash
```

### Launch the System

```bash
ros2 launch rl_control env_control_rqt.launch.py
```

This will:
- Start the env_node.
- Start the agent_node.
- Automatically open `rqt_graph` to visualise the render frames of the environment published by env_node.

While running, the nodes will log `Control‑loop latency` (end‑to‑end from observation publish to action received) as well as `Policy inference time`.


### Stop the System
Press Ctrl+C in the terminal where the launch file is running. Both nodes and rqt will shut down gracefully. In all cases env_node will stop after 100 episodes.

## Recorded Rosbag2 Files
We provide two compressed [rosbag](https://github.com/AliJnadi/RL_Assignment/tree/main/part2_ros/logs/bags) recordings:
- [successful_run](https://github.com/AliJnadi/RL_Assignment/blob/main/part2_ros/logs/bags/successful_run.zip) – the robot completes the task (object lifted to target height).
- [failed_run](https://github.com/AliJnadi/RL_Assignment/blob/main/part2_ros/logs/bags/failed_run.zip) – the robot fails because Gaussian noise (std=0.1) was added to the observation in the policy node, causing erratic actions.

To replay a bag and inspect messages:

```bash
ros2 bag play bags/successful_run
```
To view the bag info:

```bash
ros2 bag info bags/successful_run
```
(Extract the compressed archives before playing, if needed.)

## Video Demonstration
This [video](https://drive.google.com/file/d/1MDHYq4jgeMeD8rWBoQ4-R60_GDaYwB-e/view?usp=sharing) shows the robot executing the pick‑and‑place task under ROS2 closed‑loop control.
