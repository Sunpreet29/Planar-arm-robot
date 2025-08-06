from launch import LaunchDescription
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory 

def generate_launch_description():
    urdf_file_name = 'planar_robot.urdf'
    package_name = 'planar_arm_robot_description'

    urdf_path = os.path.join(
        get_package_share_directory(package_name),
        'urdf',
        urdf_file_name
    )

    rviz_config_path = os.path.join(
        get_package_share_directory(package_name),
        'rviz',
        'planar_arm_config.rviz'
    )

    with open(urdf_path, 'r') as file_handle:
        robot_description = file_handle.read()

    return LaunchDescription([
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        )
    ])
