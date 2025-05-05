# Software License Agreement (BSD)
#
# @author    Roni Kreinin <rkreinin@clearpathrobotics.com>
# @copyright (c) 2023, Clearpath Robotics, Inc., All rights reserved.
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
from ament_index_python.packages import get_package_share_directory
from clearpath_config.clearpath_config import ClearpathConfig
from clearpath_config.common.utils.yaml import read_yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    LogInfo,
    RegisterEventHandler,
    TimerAction
)
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution
)
from launch_ros.actions import PushRosNamespace, SetRemap, Node


ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='false',
                          choices=['true', 'false'],
                          description='Use sim time'),
    DeclareLaunchArgument('setup_path',
                          default_value='/etc/clearpath/',
                          description='Clearpath setup path'),
    DeclareLaunchArgument('bond_timeout', 
                          default_value='10.0',
                          description='Bond connection timeout in seconds'),
]


def launch_setup(context, *args, **kwargs):
    # Packages
    pkg_clearpath_nav2_demos = get_package_share_directory('clearpath_nav2_demos')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    # Launch Configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    setup_path = LaunchConfiguration('setup_path')
    bond_timeout = LaunchConfiguration('bond_timeout')

    # Read robot YAML
    config = read_yaml(setup_path.perform(context) + 'robot.yaml')
    # Parse robot YAML into config
    clearpath_config = ClearpathConfig(config)

    namespace = clearpath_config.system.namespace
    platform_model = clearpath_config.platform.get_platform_model()

    file_parameters = PathJoinSubstitution([
        pkg_clearpath_nav2_demos,
        'config',
        platform_model,
        'nav2.yaml'])

    launch_nav2 = PathJoinSubstitution(
      [pkg_nav2_bringup, 'launch', 'navigation_launch.py'])
    
    # Add static TF publisher to ensure required frames exist before navigation starts
    static_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_publisher_odom_base_link',
        namespace=namespace,
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    # Log info message for debugging
    log_startup = LogInfo(msg="Starting Navigation2 with increased robustness settings...")
    
    # Launch Nav2 with a delay to ensure TF frames are published
    nav2_launch = TimerAction(
        period=2.0,  # Keep the existing 2-second delay
        actions=[
            LogInfo(msg="Launching Nav2 nodes now..."),
            GroupAction([
                PushRosNamespace(namespace),
                
                # Add these remappings to prioritize vision-based scan
                SetRemap('/scan', '/scan'),  # Keep the topic name consistent
                SetRemap('/' + namespace + '/scan', '/scan'),
                
                # Existing remappings - keep as fallback
                SetRemap('/' + namespace + '/global_costmap/sensors/lidar2d_0/scan',
                        '/' + namespace + '/sensors/lidar2d_0/scan'),
                SetRemap('/' + namespace + '/local_costmap/sensors/lidar2d_0/scan',
                        '/' + namespace + '/sensors/lidar2d_0/scan'),

                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(launch_nav2),
                    launch_arguments=[
                        ('use_sim_time', use_sim_time),
                        ('params_file', file_parameters),
                        ('use_composition', 'False'),
                        ('namespace', namespace),
                        ('use_respawn', 'true'),
                        ('autostart', 'true'),
                        ('bond_timeout', bond_timeout)
                    ]
                ),
            ])
        ]
    )
    
    return [log_startup, static_tf_publisher, nav2_launch]


def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
