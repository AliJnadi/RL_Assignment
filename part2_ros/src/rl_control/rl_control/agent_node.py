import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float32MultiArray

from stable_baselines3 import SAC
from ament_index_python.packages import get_package_share_directory

from pathlib import Path
import time

import csv

import numpy as np

class AgentNode(Node):
    def __init__(self, model_path = 'resource/model.zip'):
        super().__init__('agent_node')

        # Check if the model.zip exists
        package_share = get_package_share_directory('rl_control')
        model_full_path = Path(package_share) / model_path

        if not model_full_path.exists():
            err_msg = f"Model not found: {model_full_path}"
            self.get_logger().error(err_msg)
            raise FileNotFoundError(err_msg)

        self.get_logger().info("Model loaded")
        self.model = SAC.load(str(model_full_path))

        self.inference_times = []

        self.failure_mode = False

        # Observation subscriper
        self._obs_sub = self.create_subscription(Float32MultiArray, '/observation', self._obs_cb, qos_profile_sensor_data)
        # Action Publisher
        self._action_pub = self.create_publisher(Float32MultiArray, '/action', qos_profile_sensor_data)

    def _obs_cb(self, msg):
        start = time.perf_counter()
    
        # Add noise 
        if self.failure_mode:
            obs_array = np.array(msg.data, dtype=np.float32)
            noise = np.random.normal(0.5, 0.8, size=obs_array.shape)
            data = obs_array + noise
            self.get_logger().warn("Noise added to observation (failure mode)")
        else:
            data = msg.data

        obs = {
            'observation': data[0:25],
            'achieved_goal': data[25:28],
            'desired_goal': data[28:31],
        }

        action, _ = self.model.predict(obs, deterministic=True)
        
        action_msg = Float32MultiArray()
        action_msg.data = action.tolist()

        self._action_pub.publish(action_msg)

        end = time.perf_counter()
        inference_time = end - start

        self.get_logger().info(f'Inference: {inference_time*1000:.2f}ms')

        with open('agent_diagnostics.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([inference_time*1000])

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()