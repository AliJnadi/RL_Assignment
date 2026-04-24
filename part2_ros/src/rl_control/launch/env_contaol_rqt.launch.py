from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    EnvNode = Node(
            package='rl_control',
            executable='env_node',
            name='env_node',
            output='screen',
            emulate_tty=True,
        )
    Agent_Node = Node(
            package='rl_control',
            executable='agent_node',
            name='agent_node',
            output='screen',
            emulate_tty=True,
        )
    
    RqtNode = Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_image_view',
            arguments=['/env_frame'],
            output='screen',
        )
    return LaunchDescription([
        # Environment node
        EnvNode,
        # Agent node
        Agent_Node,
        # rqt_image_view directly subscribed to /env_frame
        RqtNode,
    ])