# Author_Ragib_Rownak
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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
 
ARGUMENTS = [
    DeclareLaunchArgument('localization', default_value='false',
                          choices=['true', 'false'], description='Start rtabmap in localization mode (a map should have been already created).'),
    DeclareLaunchArgument('robot_ns', default_value='a200_1093',
                          description='Robot namespace'),
    DeclareLaunchArgument('detectron2_config', default_value='/home/administrator/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml',
                          description='Path to Detectron2 config file'),
    DeclareLaunchArgument('detectron2_weights', default_value='/home/administrator/detectron2/detectron2/model/v3.pkl',
                          description='Path to Detectron2 weights file'),
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
    rtabmap_db_path = os.path.expanduser('~/3D_Husky_ws/src/clearpath_nav2_demos/maps/rtabmap.db')

    # RTAB-Map for SLAM
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rtabmap_launch]),
        launch_arguments=[
            ('localization', LaunchConfiguration('localization')),
            ('use_sim_time', 'false'),
            ('robot_ns', robot_namespace),
            ('database_path', rtabmap_db_path),
            ('args', '--Reg/Force3DoF true --Grid3D true'),
            ('cloud_assembling', 'true'),
            # Add these critical parameters for localization mode:
            ('publish_tf_map', 'true'),
            ('map_always_update', 'true'),
            # Rest of your existing parameters
            ('rgb_topic', '/a200_1093/sensors/camera_0/color/image_rect_color'),
            ('depth_topic', '/a200_1093/sensors/camera_0/depth/image'),
            ('camera_info_topic', '/a200_1093/sensors/camera_0/color/camera_info'),
            ('approx_sync', 'true'),
            ('qos', '2'),
            ('scan_cloud_topic', '/a200_1093/sensors/camera_0/depth/color/points'),
            ('rtabmap_args', '--Optimizer/GravitySigma 0.3 --Reg/Force3DoF true --Grid3D true --Mem/IncrementalMemory true --Mem/NotLinkedNodesKept false --Db/WipeSavedLocalizationData true --Db/TargetDatabase ' + rtabmap_db_path),
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

    # Detectron2 Panoptic Segmentation Node for obstacle detection
    detectron2_node = ExecuteProcess(
        cmd=['python3', '/home/administrator/detectron2/demo_1/demo_ros.py',
             '--config-file', '/home/administrator/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml',
             '--weights', '/home/administrator/detectron2/detectron2/model/v3.pkl',
             '--confidence-threshold', '0.6',
             '--no-visualization',
             '--color-topic', '/a200_1093/sensors/camera_0/color/image',
             '--depth-topic', '/a200_1093/sensors/camera_0/depth/image'],
        output='screen'
    )

    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(rtabmap)
    ld.add_action(nav2)
    ld.add_action(detectron2_node)
    
    return ld
