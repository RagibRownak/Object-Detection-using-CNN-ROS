#
# Launch file for a200_1093 Husky with RTAB-Map SLAM and Nav2
#
# This launch file configures and starts:
#   - Husky simulation in Gazebo
#   - RTAB-Map for SLAM 
#   - Nav2 for navigation
#   - Camera navigation visualization (includes RTAB-Map viz)
#   - RViz2 with navigation configuration
#
# All components are configured to use the specified namespace.
#
# Example:
#   $ ros2 launch clearpath_nav2_demos a200_1093_demo.launch.py

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('localization', default_value='false',
                          choices=['true', 'false'], description='Start rtabmap in localization mode (a map should have been already created).'),
    DeclareLaunchArgument('world', default_value='warehouse',
                          description='Ignition World'),
    DeclareLaunchArgument('robot_ns', default_value='a200_1093',
                          description='Robot namespace'),
]

def generate_launch_description():
    # Directories
    pkg_clearpath_gz = get_package_share_directory('clearpath_gz')
    pkg_rtabmap_demos = get_package_share_directory('rtabmap_demos')
    pkg_clearpath_nav2_demos = get_package_share_directory('clearpath_nav2_demos')
    
    # Get robot namespace
    robot_namespace = LaunchConfiguration('robot_ns')
    
    # Paths
    sim_launch = PathJoinSubstitution([pkg_clearpath_gz, 'launch', 'simulation.launch.py'])
    rtabmap_launch = PathJoinSubstitution([pkg_rtabmap_demos, 'launch', 'husky', 'husky_slam2d.launch.py'])
    nav2_launch = PathJoinSubstitution([pkg_clearpath_nav2_demos, 'launch', 'nav2.launch.py'])
    camera_viz_launch = PathJoinSubstitution([pkg_clearpath_nav2_demos, 'launch', 'camera_nav_viz.launch.py'])

    # RTAB-Map database path
    rtabmap_db_path = os.path.expanduser('~/3d_mapping_ws/maps/sim_rtabmap.db')

    # Add a static transform publisher between odom and base_link to bootstrap the system
    static_transform_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_odom_base_link',
        namespace=robot_namespace,
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
    )

    # Launch simulation
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([sim_launch]),
        launch_arguments=[
            ('world', LaunchConfiguration('world')),
        ]
    )

    # RTAB-Map for SLAM - but WITHOUT its visualization component
    # We use a stripped-down version of the slam launch that doesn't include rtabmap_viz
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rtabmap_launch]),
        launch_arguments={
            # Force rtabmap_viz to false to avoid duplicates with camera_nav_viz
            'rtabmap_viz': 'false',
            'localization': LaunchConfiguration('localization'),
            'use_sim_time': 'true',
            'robot_ns': robot_namespace,
            'database_path': rtabmap_db_path,
            # If the file supports it, disable rviz too
            'rviz': 'false',
            # If the file supports it, disable any other visualization 
            'viz': 'false',
        }.items()
    )

    # Camera navigation visualization WITHOUT rtabmap_viz
    camera_viz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([camera_viz_launch]),
        launch_arguments={
            'namespace': robot_namespace,
            'use_sim_time': 'true',
            'message_queue_size': '200',
            # Explicitly disable rtabmap_viz in camera_nav_viz.launch.py
            'launch_rtabmap_viz': 'false'
        }.items()
    )

    # Navigation 2
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch]),
        launch_arguments=[
            ('setup_path', os.path.expanduser('~')+'/clearpath/'),
            ('use_sim_time', 'true'),
            ('namespace', robot_namespace),
            # If nav2 has visualization options, disable them
            ('use_rviz', 'false')
        ]
    )

    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(sim)
    ld.add_action(static_transform_odom)
    ld.add_action(rtabmap)
    ld.add_action(camera_viz)  # This ALREADY includes both rtabmap_viz and navigation visualization
    ld.add_action(nav2)
    return ld