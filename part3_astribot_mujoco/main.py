import sys
import yaml

import mujoco
from mujoco import viewer

import time

from Robot import Robot
import numpy as np

def main(yaml_file):
    # Load YAML configuration
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract model path
    model_path = config.get('model_path')
    if not model_path:
        print("Error: 'model_path' not found in YAML file.")
        sys.exit(1)

    try:
        # Load MuJoCo model and create data structure
        robot = Robot(config)

        # Launch interactive viewer
        print("\nVisualization started")
        launch_viewer(robot)

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    finally:
        # Clean up
        if 'robot' in locals():
            robot.close()

def launch_viewer(robot):
    """Launch interactive viewer that runs until ESC is pressed"""
    # Option 1: Using mujoco.viewer (simplest, modern approach)
    with viewer.launch_passive(
        robot.model, 
        robot.data,
        show_left_ui=False,
        show_right_ui=False
    ) as v:
        
        # Set initial camera
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance = 5.0
        v.cam.azimuth = 45
        v.cam.elevation = -20

        def make_trajectory(a, b, num):
            forward = np.linspace(a, b, num)
            backward = np.linspace(b, a, num)[1:]
            result = np.concatenate([forward, backward]).tolist()
            return result

        # Reset simulation time
        robot.data.time = 0.0
        control_steps = 200

        # Smooth trajectory
        trajectory_joint_space = {
            "astribot_torso_joint_1": make_trajectory(0, +np.pi/4, control_steps),
            "astribot_torso_joint_2": make_trajectory(0, -np.pi/2, control_steps),
            "astribot_torso_joint_3": make_trajectory(0, +np.pi/4, control_steps),
                
            'astribot_gripper_left_joint_L1': make_trajectory(0, 10, control_steps),
            'astribot_gripper_right_joint_L1': make_trajectory(0, 10, control_steps),

            'astribot_arm_left_joint_1': make_trajectory(0, -np.pi/2, control_steps),
            'astribot_arm_right_joint_1': make_trajectory(0, +np.pi/2, control_steps),

            'astribot_arm_left_joint_2': make_trajectory(0, -np.pi/2, control_steps),
            'astribot_arm_right_joint_2': make_trajectory(0, -np.pi/2, control_steps),

            'astribot_arm_left_joint_3': make_trajectory(0, np.pi/2, control_steps),
            'astribot_arm_right_joint_3': make_trajectory(0, -np.pi/2, control_steps),

            'astribot_arm_left_joint_4': make_trajectory(0, np.pi/2, control_steps),
            'astribot_arm_right_joint_4': make_trajectory(0, np.pi/2, control_steps),

            'astribot_arm_left_joint_5': make_trajectory(0, np.pi, control_steps),
            'astribot_arm_right_joint_5': make_trajectory(0, -np.pi, control_steps),

            'astribot_arm_left_joint_6': make_trajectory(0, -np.pi/2, control_steps),
            'astribot_arm_right_joint_6': make_trajectory(0, -np.pi/2, control_steps),

            'astribot_arm_left_joint_7': make_trajectory(0, -np.pi/2, control_steps),
            'astribot_arm_right_joint_7': make_trajectory(0, np.pi/2, control_steps),
            'astribot_chassis_x': make_trajectory(0, 1.0, control_steps),
        }
        
        trajectory_joint_space_collide = {
            'astribot_arm_left_joint_1': make_trajectory(0, -np.pi, control_steps),
            'astribot_arm_right_joint_1': make_trajectory(0, np.pi, control_steps),
        }
        # save_looping_gif(robot, trajectory_joint_space, total_steps, "robot_animation.gif", fps=30)
        
        # Main simulation loop
        total_steps = control_steps * 2 - 1 # Forward + backward - 1
        idx = 0
        idx_c = 0
        while v.is_running():
            if idx < total_steps:
                for name, value in trajectory_joint_space.items():
                    robot.control_actuator(name, value[idx])
                idx += 1
            else:
                # Show the contact points when arms touch the head
                if idx_c < total_steps:
                    for name, value in trajectory_joint_space_collide.items():
                        robot.control_actuator(name, value[idx_c])
                idx_c += 1
                

            robot.step_simulation()
            v.sync()

            # if int(robot.data.time * 10) % 10 == 0 and robot.data.time > 0:
            #     for group_name, names in robot.joint_names_group.items():
            #         print(f"{group_name}:")
            #         for name in names:
            #             joint_pos = robot.get_joint_position_by_name(robot, name)
            #             print(f"     {name}: {joint_pos:.2f}")
            #     print("-"*25)

            # Check if there are any contacts
            if robot.data.ncon > 0:
                print(f"Number of contacts: {robot.data.ncon}")
                
                # Loop through active contacts
                for i in range(robot.data.ncon):
                    contact = robot.data.contact[i]

                    # Get geometry names
                    geom1_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                    geom2_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                    
                    print(f"Collision between {geom1_name} and {geom2_name}")
                    print(f"Position: {contact.pos}")
            
            time.sleep(0.01)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])