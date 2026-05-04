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

### 1. Relative Path Resolution
- **Problem:** The MJCF file referenced meshes using relative paths, which caused failures when executed from different directories.
- **Solution:** Changed the working directory to the MJCF file location before loading:
    ```python
    os.chdir('astribot_descriptions/mjcf/astribot_s1_mjcf')
    ```

### 2. Git LFS Pointer Files
- **Problem:** Mesh files consisted of Git LFS pointers rather than actual 3D data, triggering the error:
    ```bash
    Error: "at least 4 vertices required"
    ```
- **Solution:** Ran `git lfs pull` to download the actual mesh files (approximately 33MB each). Alternatively, download and manually add them to the `astribot_descriptions/urdf/astribot_s1_urdf/meshes/obj` folder.

### 3. Viewer Compatibility (macOS)
- On macOS, the code must be executed using `mjpython`, which is automatically installed with MuJoCo.

## Controlling the Robot

The joint command function is `control_actuator(joint_name, command)` from the `Robot` class, where:
- `joint_name` specifies the target joint (available joint names are defined in `config.yaml`)
- `command` is the desired control value (position, angle, torque, etc.)

The function automatically respects the joint limits defined in the same `config.yaml` file.

## Configuration File (`config.yaml`)

The `[config.yaml](https://github.com/AliJnadi/RL_Assignment/tree/main/part3_astribot_mujoco/config.yaml)` file serves as the central configuration hub for the simulation.

### Key Configuration Sections

| Section | Description |
|---------|-------------|
| `robot_list` | Defines all robot body parts |
| `joint_names_list` | Complete list of controllable joints |
| `joint_names_group` | Groups joints by functional body part |
| `actuators` | Maps each joint to an actuator ID and defines its movement limits |
| `gravity_compensation` | Enables/disables gravity compensation for smoother control |
| `mode` | Simulation mode (`"human"` for real-time, other options available) |
| `width` / `height` | Viewer window dimensions |
| `camera` | Preset camera position (`'close'`, `'far'`, etc.) |

## Demonstrations