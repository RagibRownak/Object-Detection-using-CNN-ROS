#!/usr/bin/env python3
#Author_Ragib_Rownak
# Software License Agreement (BSD)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Clearpath Robotics nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Arguments
    arg_namespace = DeclareLaunchArgument(
        'namespace',
        default_value='a200_1093',
        description='Robot namespace'
    )

    arg_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    arg_message_queue_size = DeclareLaunchArgument(
        'message_queue_size',
        default_value='200',
        description='Size of the RViz message filter queue'
    )
    
    # Launch Configurations
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    message_queue_size = LaunchConfiguration('message_queue_size')
    
    # Find packages
    pkg_clearpath_viz = FindPackageShare('clearpath_viz')
    pkg_rtabmap_ros = FindPackageShare('rtabmap_ros')
    
    # Optimized FFMPEG decoder with lower latency settings
    ffmpeg_decoder = Node(
        package='image_transport',
        executable='republish',
        name='ffmpeg_decoder',
        arguments=['ffmpeg', 'raw'],
        remappings=[
            ('in', ['/', namespace, '/sensors/camera_0/intel_realsense/color/image_raw/ffmpeg_panoptic']),
            ('out', ['/', namespace, '/sensors/camera_0/color/decoded_image'])
        ],
        parameters=[
            {'use_sim_time': use_sim_time},
            {'reliability': 'best_effort'},   # Choose best effort for lower latency
            {'history_depth': 1},             # Only keep latest frame
            {'deadline_period': 0.1},         # Set deadline in seconds
            {'buffer_size': 1},               # Minimum buffer size
        ],
        output='screen'
    )
    
    # Direct image relay (skipping topic_tools to reduce latency)
    direct_image_transport = Node(
        package='image_transport',
        executable='republish',
        name='direct_relay',
        arguments=['raw', 'raw'],
        remappings=[
            ('in', ['/', namespace, '/sensors/camera_0/color/decoded_image']),
            ('out', ['/', namespace, '/sensors/camera_0/color/image_panoptic'])
        ],
        parameters=[
            {'use_sim_time': use_sim_time},
            {'reliability': 'best_effort'},
            {'history_depth': 1},
        ],
        output='screen'
    )
    
    # Include the standard navigation visualization launch
    nav_viz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_clearpath_viz, 'launch', 'view_navigation.launch.py'])
        ]),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'message_queue_size': message_queue_size
        }.items()
    )
    
    # Add RTAB-Map visualization
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'frame_id': 'base_link'},
            {'odom_frame_id': 'odom'},
            {'wait_for_transform': 0.2},
            {'approx_sync': True},
            {'queue_size': 10}
        ],
        remappings=[
            ('grid_map', 'map'),
            ('grid_prob_map', 'map_prob'),
            ('scan', 'sensors/lidar2d_0/scan'),
            ('rgb/image', 'sensors/camera_0/color/image'),
            ('rgb/camera_info', 'sensors/camera_0/color/camera_info'),
            ('rgbd_image', 'rgbd_image'),
            ('odom', 'odom')
        ],
        output='screen'
    )
    
    # Create launch description
    ld = LaunchDescription()
    ld.add_action(arg_namespace)
    ld.add_action(arg_use_sim_time)
    ld.add_action(arg_message_queue_size)
    ld.add_action(ffmpeg_decoder)
    ld.add_action(direct_image_transport)
    ld.add_action(nav_viz)
    ld.add_action(rtabmap_viz_node)
    
    return ld
