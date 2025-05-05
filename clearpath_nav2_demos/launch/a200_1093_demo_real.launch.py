#
# Launch file for a200_1093 real Husky with RTAB-Map SLAM and Nav2
#
# This launch file configures and starts:
#   - RTAB-Map for SLAM
#   - Nav2 for navigation
#
# All components are configured to use the specified namespace.
#
# Example:
#   $ ros2 launch clearpath_nav2_demos a200_1093_demo_real.launch.py

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition

ARGUMENTS = [
    DeclareLaunchArgument('rtabmap_viz', default_value='true',
                          choices=['true', 'false'], description='Start rtabmap_viz.'),
    DeclareLaunchArgument('localization', default_value='false',
                          choices=['true', 'false'], description='Start rtabmap in localization mode (a map should have been already created).'),
    DeclareLaunchArgument('robot_ns', default_value='a200_1093',
                          description='Robot namespace'),
]

def generate_launch_description():
    # Directories
    pkg_rtabmap_demos = get_package_share_directory('rtabmap_demos')
    pkg_clearpath_nav2_demos = get_package_share_directory('clearpath_nav2_demos')
    
    # Get robot namespace
    robot_namespace = LaunchConfiguration('robot_ns')
    
    # Paths
    rtabmap_launch = PathJoinSubstitution([pkg_rtabmap_demos, 'launch', 'husky', 'husky_slam2d.launch.py'])
    nav2_launch = PathJoinSubstitution([pkg_clearpath_nav2_demos, 'launch', 'nav2.launch.py'])
    
    # Database path for RTAB-Map
    rtabmap_db_path = os.path.expanduser('~/3d_mapping_ws/maps/rtabmap.db')

    # RTAB-Map for SLAM
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rtabmap_launch]),
        launch_arguments=[
            ('rtabmap_viz', LaunchConfiguration('rtabmap_viz')),
            ('localization', LaunchConfiguration('localization')),
            # For real robot, use_sim_time should be false
            ('use_sim_time', 'false'),
            ('robot_ns', robot_namespace),
            ('database_path', rtabmap_db_path),
        ]
    )

    # Navigation 2
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch]),
        launch_arguments=[
            ('setup_path', os.path.expanduser('~')+'/clearpath/'),
            # For real robot, use_sim_time should be false
            ('use_sim_time', 'false'),
            ('namespace', robot_namespace)
        ]
    )

    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(rtabmap)
    ld.add_action(nav2)
    
    return ld