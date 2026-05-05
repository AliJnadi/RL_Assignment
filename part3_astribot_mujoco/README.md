# Astribot Mujoco: Astribot S1 - MuJoCo Simulation & Control

Integration and demonstration control of the Astribot S1 semi-humanoid robot within the MuJoCo physics simulator.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AliJnadi/RL_Assignment
```

2. Navigate to the description folder:
```bash
cd part3_astribot_mujuco/astribot_descriptions
```

3. Download the actual mesh files:
```bash
git lfs pull  
cd ..
```

4. Download required modules
```bash
pip install -r requirements.txt
```

5. Run `main.py`:
   - **Non-macOS systems:**
     ```bash 
     python3 main.py config.yaml
     ```
   - **macOS:**
     ```bash 
     mjpython main.py config.yaml 
     ```

## Issues Encountered & Solutions

### 1. Git LFS Pointer Files
- **Problem:** Mesh files consisted of Git LFS pointers rather than actual 3D data, triggering the error:
    ```bash
    Error: "at least 4 vertices required"
    ```
- **Solution:** Run `git lfs pull` to download the actual mesh [files](https://github.com/AliJnadi/RL_Assignment/tree/main/part3_astribot_mujoco/astribot_descriptions/urdf/astribot_s1_urdf/meshes/obj) (approximately 33MB each). Alternatively, manually download and add them to the `astribot_descriptions/urdf/astribot_s1_urdf/meshes/obj` folder.

### 2. Viewer Compatibility (macOS)
- On macOS, the code must be executed using `mjpython`, which is automatically installed with MuJoCo.

### 3. Collision seems not to be working
- You are using the original repository. In this repo, the XML files have been modified to include collision geometries while preserving the graphical resolution in the simulator.

### 4. Changing joint limits
If you want to add or change **joint limits**, edit the [`config.yaml`](https://github.com/AliJnadi/RL_Assignment/tree/main/part3_astribot_mujoco/config.yaml) file.  
Navigate to:

```yaml
actuators:
  name: <actuator_name>
  id: joint_id
  limits: [lower_limit, upper_limit]
```

### 5. Modify collosions geometry
If you want to modify **collision geometries**, look for lines like the following in the XML files.  
Start from the [main XML file](https://github.com/AliJnadi/RL_Assignment/blob/main/part3_astribot_mujoco/astribot_descriptions/mjcf/astribot_s1_mjcf/astribot_s1_with_gripper.xml) and follow the includes.

Example line to change:

```xml
<geom pos="-0.18 0.01 0" euler="0 1.57 0" type="cylinder" size="0.09 0.175" group="1" rgba="0 0 0 0" contype="1" conaffinity="1"/>
```

Adjust `pos`, `size`, `radius` (for spheres) or half‑length (second value in `size` for cylinders) as needed.

## Controlling the Robot

The joint command function is `control_actuator(joint_name, command)` from the `Robot` class, where:
- `joint_name` specifies the target joint (available joint names are defined in `config.yaml`)
- `command` is the desired control value (position, angle, torque, etc.)

The function automatically respects the joint limits defined in the same `config.yaml` file.

## Configuration File

The [`config.yaml`](https://github.com/AliJnadi/RL_Assignment/tree/main/part3_astribot_mujoco/config.yaml) file serves as the central configuration hub for the simulation.

### Key Configuration Sections

| Section | Description |
|---------|-------------|
|`model_path` | Path to model xml file|
| `robot_list` | Defines all robot body parts |
| `joint_names_list` | Complete list of controllable joints |
| `joint_names_group` | Groups joints by functional body part |
| `actuators` | Maps each joint to an actuator ID and defines its movement limits |
| `gravity_compensation` | Enables/disables gravity compensation for smoother control |
| `mode` | Simulation mode (`"human"` for real-time, other options available) |
| `width` / `height` | Viewer window dimensions |
| `camera` | Preset camera position (`'close'`, `'far'`, etc.) |

## Demonstrations
<img width="960" height="663" alt="viewer" src="https://github.com/user-attachments/assets/7ef272dc-3cbd-44ef-afed-9403c6833e68" />
