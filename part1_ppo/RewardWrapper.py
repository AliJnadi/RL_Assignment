import gymnasium as gym
import numpy as np

class RewardWrapperSparse(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += 1
        return obs, reward, terminated, truncated, info

class RewardWrapperDense(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._was_lifted = False

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        goal = obs['desired_goal']
        obj = obs['achieved_goal']
        ee = obs['observation'][0:3]

        # gripper: 0 (closed), 0.05 (open)
        gripper_open = obs['observation'][10]

        # distances
        d_ee_obj = np.linalg.norm(ee - obj)
        d_obj_goal = np.linalg.norm(obj - goal)

        # object height
        obj_z = obj[2]

        # Hyperparameters
        initial_obj_z = 0.42
        lift_threshold = 0.03
        grasp_threshold = 0.05

        # Base shaping
        reach_reward = -d_ee_obj
        place_reward = -d_obj_goal
        action_penalty = -np.linalg.norm(action)

        # Gripper shaping with a distance gate
        if d_ee_obj > grasp_threshold:
            gripper_reward = +1.0 * gripper_open          # open while reaching
        else:
            gripper_reward = +2.0 * (0.05 - gripper_open) # close when near object

        # Lift success bonus (sparse)
        lift_success_bonus = 0.0
        if not self._was_lifted and obj_z > initial_obj_z + lift_threshold:
            lift_success_bonus = 10.0
            self._was_lifted = True

        # Continuous lift shaping
        lift_shaping = np.clip((obj_z - initial_obj_z) / 0.05, 0, 1) * 0.5

        reward = (
            1.0 * reach_reward +
            2.0 * gripper_reward +
            0.5 * lift_shaping +           
            lift_success_bonus +           
            3.0 * place_reward +
            0.01 * action_penalty
        )

        # Final success condition
        
        if d_obj_goal < 0.05:
            reward += 10
            terminated = True
            info['is_success'] = 1
        else:
            info['is_success'] = 0

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self._was_lifted = False
        return super().reset(seed=seed, options=options)