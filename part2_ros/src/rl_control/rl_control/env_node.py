import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from cv_bridge import CvBridge

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ament_index_python.packages import get_package_share_directory

from pathlib import Path

import numpy as np
import csv

class EnvNode(Node):
    def __init__(self, env_path='resource/env.pkl', normalised=True,):
        super().__init__('env_node')

        self.get_logger().info("Initializing EnvNode...")

        # Register and create environment
        self.get_logger().info("Registering Gymnasium robotics environments.")
        gym.register_envs(gymnasium_robotics)

        self.get_logger().info("Creating FetchPickAndPlace-v4 environment.")
        env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array")
        self.get_logger().debug("Base environment created successfully.")

        # Wrap in DummyVecEnv (needed for VecNormalize)
        env = DummyVecEnv([lambda: env])
        self.get_logger().info("Environment wrapped in DummyVecEnv (1 env).")

        # Load normalization if requested
        if normalised:
            package_share = get_package_share_directory('rl_control')
            env_full_path = Path(package_share) / env_path

            if not env_full_path.exists():
                err_msg = f"Normalization model not found: {env_full_path}"
                self.get_logger().error(err_msg)
                raise FileNotFoundError(err_msg)

            self.get_logger().info(f"Loading VecNormalize from: {env_full_path}")
            self.env = VecNormalize.load(str(env_full_path), env)
            self.env.training = False
            self.env.norm_reward = False
            self.get_logger().info("VecNormalize loaded and set to eval mode.")
        else:
            self.get_logger().info("Skipping normalization, running environment raw.")
            self.env = env

        # For joint space
        self.sim_env = self.env.envs[0].unwrapped

        self.episode_latencies = []          # seconds, per episode
        self.episode_success = None          # will hold the success flag of current episode
        self.all_successes = []              # across all episodes
        self._new_action = False
        self.action = None

        # Image publisher
        self.get_logger().info("Setting up image publisher on /env_frame...")
        self.bridge = CvBridge()
        self._render_pub = self.create_publisher(Image, "/env_frame", qos_profile_sensor_data)

        # Joint states publisher
        self.get_logger().info("Setting up joints publisher on /joint_states...")
        self._joint_pub = self.create_publisher(JointState, "/joint_states", qos_profile_sensor_data)
        # Object pose publisher
        self.get_logger().info("Setting up object pose publisher on /object_pose...")
        self._obj_pose_pub = self.create_publisher(PoseStamped, "/object_pose", qos_profile_sensor_data)
        # end‑effector pose publisher
        self.get_logger().info("Setting up gripper pose publisher on /gripper_pose...")
        self._ee_pose_pub = self.create_publisher(PoseStamped, "/gripper_pose", qos_profile_sensor_data)

        # Obs publisher with a fixed frequency
        self.get_logger().info("Setting up observation publisher on /observation...")
        self._obs_pub = self.create_publisher(Float32MultiArray, '/observation', qos_profile_sensor_data)

        # Action subscriber
        self.get_logger().info("Setting up action subscriper on /action...")
        self._action_sub = self.create_subscription(Float32MultiArray, '/action', self._action_cb, qos_profile_sensor_data)

        # Reset environment and obtain initial observation
        self.get_logger().info("Resetting environment...")
        self.obs = self.env.reset()
        self.episode_count = 1

        # Render first frame
        self.get_logger().info("Rendering initial frame...")
        self.frames = self.env.render()
        # DummyVecEnv returns a list of frames (one per env)
        frame = self.frames[0] if isinstance(self.frames, list) else self.frames
        self.get_logger().debug(f"Frame shape: {frame.shape if hasattr(frame, 'shape') else 'unknown'}")

        # Publish render frames and sensor data at 50Hz
        self._sensor_rate = 50
        self._sensor_timer = self.create_timer(1.0/ self._sensor_rate, self._sensor_pub_tcb)
        self.get_logger().info("Sensore timer created at 100 Hz for frames and state publishing.")
        
        # Publish observation at 25Hz
        self._controller_rate = 25
        self._controller_timer = self.create_timer(1.0/ self._controller_rate, self._obs_pub_cb)
        self.get_logger().info("Conrol timer created at 25 Hz for observation publishing.")

        self.get_logger().info("EnvNode initialization complete.")

    def _frame_pub_cb(self):
        """Timer callback: render and publish current frame."""
        try:
            frame = self.frames[0] if isinstance(self.frames, list) else self.frames

            if frame is not None:
                # gym renders in RGB, cv_bridge expects matching encoding
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
                self._render_pub.publish(img_msg)
            else:
                self.get_logger().warn("render() returned None frame.")
        except Exception as e:
            self.get_logger().error(f"Error in _render_pub_tcb: {e}")

    def _state_pub_cb(self):
        # Joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = "world"  # or "base_link"
        # Fetch arm: 7 joints (indices 0..6 in qpos)
        joint_msg.name = [
            "shoulder_pan_joint", "shoulder_lift_joint",
            "upperarm_roll_joint", "elbow_flex_joint",
            "forearm_roll_joint", "wrist_flex_joint",
            "wrist_roll_joint"
        ]
        # joint_msg.position = self.sim_env.data.qpos[:7].tolist()
        joint_msg.position = self.sim_env.data.qpos[:7].tolist()
        self._joint_pub.publish(joint_msg)

        # Object pose
        achieved_goal = self.obs['achieved_goal'][0]  # array of 3 elements
        obj_msg = PoseStamped()
        obj_msg.header.stamp = self.get_clock().now().to_msg()
        obj_msg.header.frame_id = "world"
        obj_msg.pose.position = Point(x=achieved_goal[0], y=achieved_goal[1], z=achieved_goal[2])
        obj_msg.pose.orientation = Quaternion(w=1.0)
        self._obj_pose_pub.publish(obj_msg)

        # End‑effector pose
        obs_vector = self.obs['observation'][0]  # 1D array
        ee_pos = obs_vector[0:3]  # your specified indices
        ee_msg = PoseStamped()
        ee_msg.header.stamp = obj_msg.header.stamp
        ee_msg.header.frame_id = "world"
        ee_msg.pose.position = Point(x=ee_pos[0], y=ee_pos[1], z=ee_pos[2])
        ee_msg.pose.orientation = Quaternion(w=1.0)
        self._ee_pose_pub.publish(ee_msg)

    def _obs_pub_cb(self):
        """Send the latest observation at the timer rate."""
        msg = Float32MultiArray()
        # self.obs is a dict with shape (1, dim) from DummyVecEnv
        data = self.obs['observation'][0].tolist()
        [data.append(v) for v in self.obs['achieved_goal'][0].tolist()]
        [data.append(v) for v in self.obs['desired_goal'][0].tolist()]
        
        msg.data = data

        self._obs_pub.publish(msg)
        self.last_obs_time = self.get_clock().now()
        
        self.get_logger().debug('Published observation')

    def _action_cb(self, msg):
        if hasattr(self, 'last_obs_time'):
            latency = (self.get_clock().now() - self.last_obs_time).nanoseconds * 1e-9
            self.episode_latencies.append(latency)
            self.get_logger().debug(f'Control latency: {latency:.4f}s')

            with open('latency_diagnostics.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([latency])

        action = np.array(msg.data, dtype=np.float32)
        self.action = np.expand_dims(action, axis=0)
        # Apply the new action
        self.obs, _, done, info = self.env.step(self.action)

        if done[0]:
            self.get_logger().info("Episode finished, resetting.")
            # Retrieve success info (if available)
            self.episode_success = info[0]['is_success']
            self.all_successes.append(self.episode_success)

            # Compute and log episodic diagnostics
            self.print_episode_diagnostics()
            self.episode_latencies = []

            with open('env_diagnostics_success.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_count, self.episode_success])

            if self.episode_count == 101:
                self.get_logger().info("Excution finished")
                rclpy.shutdown()

            self.obs = self.env.reset()

            self.episode_count += 1
        


    def _sensor_pub_tcb(self):
        self.frames = self.env.render()
        self._frame_pub_cb()
        self._state_pub_cb()

    def print_episode_diagnostics(self):
        if self.episode_latencies:
            mean_lat = np.mean(self.episode_latencies)
            p95_lat = np.percentile(self.episode_latencies, 95)
            self.get_logger().info(
                f'Episode {len(self.all_successes)}: '
                f'Latency mean={mean_lat*1000:.2f}ms, p95={p95_lat*1000:.2f}ms'
            )
        if self.episode_success is not None:
            self.get_logger().info(f'  Success: {self.episode_success}')
            


def main(args=None):
    rclpy.init(args=args)
    node = EnvNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()