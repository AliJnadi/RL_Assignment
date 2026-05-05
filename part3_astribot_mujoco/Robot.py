import os
import yaml

import mujoco
import numpy as np

class Robot():
    def __init__(self, param):
        self.height = param.get('height', 460)
        self.width = param.get('width', 640)
        self.camera = param.get('camera', 'close')
        
        self.model_path = param.get('model_path', '')
        
        self.collisions = param.get('collision', '')
        
        self.setup_mujoco_model_and_data()
        self.setup_renderer()
        # Get useful indices
        self.n_joints = self.model.njnt
        self.n_actuators = self.model.nu
        self.joint_names_list = param.get('joint_names_list', [])
        self.joint_names_group = param.get('joint_names_group', [])

        # Extract actators id and limits
        self.actuators = param.get('actuators', None)
        assert self.actuators is not None, "Error: No actuators informations in config file"

        # Print model info for verification
        print(f"Model Info:")
        print(f"  - Bodies: {self.model.nbody}")
        print(f"  - Joints: {self.model.njnt}")
        print(f"  - Actuators: {self.model.nu}")
        print(f"  - Degrees of freedom: {self.model.nv}")

    def setup_mujoco_model_and_data(self):
        print("Setup mujoco model and data")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(BASE_DIR, self.model_path)

        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)

        # Create data
        self.data = mujoco.MjData(self.model)

    def setup_renderer(self):
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        
    def render(self):
        self.renderer.update_scene(self.data)
        return self.renderer.render()
    
    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()
    
    def get_joint_angles(self):
        """Get current joint angles"""
        return self.data.qpos.copy()
    
    def step_simulation(self):
        """Step the simulation forward"""
        mujoco.mj_step(self.model, self.data)
        
    def reset(self):
        """Reset simulation to default pose"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    
    def control_actuator(self, actuator_name, command):
        """
        Control any actuator by name with automatic limit checking
        
        Args:
            actuator_name: Name string of the actuator
            command: Control signal (automatically clamped to limits)
        """
        actuator = self.actuators[actuator_name]
        id = actuator['id']
        ctrl_range = actuator['limits']
        # Automatically clamp to control limits if they exist
        
        if len(ctrl_range) != 0:
            command_cliped = np.clip(command, ctrl_range[0], ctrl_range[1])
            if command != command_cliped:  # Check if modified
                command = command_cliped
                print(f"Control clamped to [{ctrl_range[0]:.2f}, {ctrl_range[1]:.2f}]")
        
        # Apply control
        self.data.ctrl[id] = command
        return True
    
    def get_joint_position_by_name(self, joint_name):
        """Get position of a specific joint by name"""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return None
        
        qpos_addr = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_addr]