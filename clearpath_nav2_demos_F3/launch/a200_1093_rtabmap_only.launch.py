# Author_Ragib_Rownak
# Launch file for a200_1093 real Husky with RTAB-Map SLAM only
#
# This launch file configures and starts:
#   - RTAB-Map for SLAM
#
# All components are configured to use the specified namespace.
#
# Example:
#   $ ros2 launch clearpath_nav2_demos a200_1093_rtabmap_only.launch.py

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

ARGUMENTS = [
    DeclareLaunchArgument('localization', default_value='false',
                          choices=['true', 'false'], description='Start rtabmap in localization mode (a map should have been already created).'),
    DeclareLaunchArgument('robot_ns', default_value='a200_1093',
                          description='Robot namespace'),
]

def generate_launch_description():
    # Directories
    pkg_rtabmap_demos = get_package_share_directory('rtabmap_demos')
    
    # Get robot namespace
    robot_namespace = LaunchConfiguration('robot_ns')
    
    # Paths
    rtabmap_launch = PathJoinSubstitution([pkg_rtabmap_demos, 'launch', 'husky', 'husky_slam2d.launch.py'])
    
    # Database path for RTAB-Map
    rtabmap_db_path = os.path.expanduser('~/3D_Husky_ws/src/clearpath_nav2_demos/maps/rtabmap.db')

    # RTAB-Map for SLAM
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rtabmap_launch]),
        launch_arguments=[
            ('localization', LaunchConfiguration('localization')),
            # For real robot, use_sim_time should be false
            ('use_sim_time', 'false'),
            ('robot_ns', robot_namespace),
            ('database_path', rtabmap_db_path),
            ('args', '--Reg/Force3DoF true --Grid3D true'),
            ('cloud_assembling', 'true'),
            ('rgb_topic', '/a200_1093/sensors/camera_0/color/image_rect_color'),
            ('depth_topic', '/a200_1093/sensors/camera_0/depth/image'),
            ('camera_info_topic', '/a200_1093/sensors/camera_0/color/camera_info'),
            ('approx_sync', 'true'),
            ('qos', '2'),
            ('scan_cloud_topic', '/a200_1093/sensors/camera_0/depth/color/points'),
            ('rtabmap_args', '--Optimizer/GravitySigma 0.3 --Reg/Force3DoF true --Grid3D true --Mem/IncrementalMemory true --Mem/NotLinkedNodesKept false --Db/WipeSavedLocalizationData true --Db/TargetDatabase ' + rtabmap_db_path),
        ]
    )

    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(rtabmap)
    
    return ld