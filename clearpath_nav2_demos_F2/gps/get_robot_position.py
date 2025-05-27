#!/usr/bin/env python3
# filepath: /home/ragib/clearpath_ws_2/src/clearpath_nav2_demos/gps/get_robot_position.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformListener, Buffer
import math
import sys
import time

class RobotPositionGetterFixed(Node):
    def __init__(self):
        super().__init__('robot_position_getter_fixed')
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Robot namespace
        self.robot_ns = "a200_1093"
        
        # Flag to ensure we only get one reading
        self.position_found = False
        
        # Subscribe to AMCL pose (for localization mode)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f'/{self.robot_ns}/amcl_pose', 
            self.amcl_callback,
            10
        )
        
        # Enhanced odometry sources for both sim and real robot
        self.odom_subs = [
            # Standard platform odometry (works for both sim and real)
            self.create_subscription(Odometry, f'/{self.robot_ns}/platform/odom', 
                                   lambda msg: self.odom_callback(msg, "Platform Odom"), 10),
            self.create_subscription(Odometry, f'/{self.robot_ns}/platform/odom/filtered', 
                                   lambda msg: self.odom_callback(msg, "Filtered Odom"), 10),
            
            # RTAB-Map ICP odometry (high accuracy)
            self.create_subscription(Odometry, f'/{self.robot_ns}/icp_odom', 
                                   lambda msg: self.odom_callback(msg, "ICP Odom"), 10),
                                   
            # Main odometry topic
            self.create_subscription(Odometry, f'/{self.robot_ns}/odom', 
                                   lambda msg: self.odom_callback(msg, "Main Odom"), 10),
            
            # Additional real robot sources (if available)
            self.create_subscription(Odometry, f'/{self.robot_ns}/gps/odom', 
                                   lambda msg: self.odom_callback(msg, "GPS Odom"), 10),
            self.create_subscription(Odometry, f'/{self.robot_ns}/ekf/odom', 
                                   lambda msg: self.odom_callback(msg, "EKF Odom"), 10),
        ]
        
        # Timer to get position via TF2 (works for both SLAM and localization)
        self.timer = self.create_timer(2.0, self.get_position_from_tf)
        
        self.get_logger().info("Getting robot position in map frame...")
        self.get_logger().info(f"Using robot namespace: {self.robot_ns}")
        self.get_logger().info("Trying odometry sources and TF...")
        
        # Check if running on real robot vs simulation
        self.check_robot_type()
        
    def check_robot_type(self):
        """Detect if running on real robot or simulation"""
        topics = self.get_topic_names_and_types()
        topic_names = [topic for topic, _ in topics]
        
        if '/clock' in topic_names:
            self.get_logger().info("🤖 Detected: Simulation environment")
        else:
            self.get_logger().info("🔧 Detected: Real robot environment")
    
    # ADD: Odometry callback method
    def odom_callback(self, msg, source):
        """Get position from odometry (WORKING METHOD)"""
        if self.position_found:
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Get orientation and convert to yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Print results
        self.print_results(x, y, z, yaw, source, msg.header.frame_id)
        
        self.position_found = True
        self.shutdown_node()

    def amcl_callback(self, msg):
        """Get position from AMCL pose (localization mode)"""
        if self.position_found:
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Get orientation and convert to yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Print results
        self.print_results(x, y, z, yaw, "AMCL (Localization)", msg.header.frame_id)
        
        self.position_found = True
        self.shutdown_node()
    
    def get_position_from_tf(self):
        """Get position using TF2 (works for both SLAM and localization)"""
        if self.position_found:
            return
            
        # Try different frame combinations
        frame_combinations = [
            ('map', 'base_link'),
            (f'{self.robot_ns}/map', f'{self.robot_ns}/base_link'),
            ('map', f'{self.robot_ns}/base_link'),
            (f'{self.robot_ns}/map', 'base_link'),
            (f'{self.robot_ns}/odom', f'{self.robot_ns}/base_link'),
            ('odom', 'base_link'),
            ('odom', f'{self.robot_ns}/base_link'),
        ]
        
        for target_frame, source_frame in frame_combinations:
            try:
                self.get_logger().debug(f"Trying transform: {target_frame} -> {source_frame}")
                
                # Get transform
                transform = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time())
                
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                z = transform.transform.translation.z
                
                # Get orientation and convert to yaw
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
                # Determine mode based on available topics
                mode = self.determine_mode()
                
                # Print results
                self.print_results(x, y, z, yaw, f"TF2 ({mode})", target_frame)
                
                self.position_found = True
                self.shutdown_node()
                return
                
            except Exception as e:
                continue
        
        self.get_logger().debug("Still waiting for valid transform...")
    
    def determine_mode(self):
        """Determine if we're in SLAM or localization mode"""
        try:
            topic_list = self.get_topic_names_and_types()
            topic_names = [topic for topic, _ in topic_list]
            
            amcl_topics = [topic for topic in topic_names if 'amcl' in topic]
            rtabmap_topics = [topic for topic in topic_names if 'rtabmap' in topic]
            
            if amcl_topics:
                return "Localization (AMCL)"
            elif rtabmap_topics:
                return "SLAM (RTAB-Map)"
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    def print_results(self, x, y, z, yaw, source, frame_id=""):
        """Print the robot position results"""
        yaw_degrees = math.degrees(yaw)
        
        print(f"\n=== Robot Position ===")
        print(f"Source: {source}")
        if frame_id:
            print(f"Reference Frame: {frame_id}")
        print(f"Position:")
        print(f"  X: {x:.6f} meters")
        print(f"  Y: {y:.6f} meters") 
        print(f"  Z: {z:.6f} meters")
        print(f"Orientation:")
        print(f"  Yaw: {yaw:.6f} radians ({yaw_degrees:.2f} degrees)")
        print(f"2D Coordinates: ({x:.3f}, {y:.3f})")
        print("======================\n")
    
    def shutdown_node(self):
        """Shutdown the node properly"""
        self.timer.cancel()
        rclpy.shutdown()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Get robot position in map frame (works for sim and real robot)')
    parser.add_argument('--timeout', type=float, default=15.0,
                       help='Timeout in seconds (default: 15.0)')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    node = RobotPositionGetterFixed()
    
    try:
        start_time = time.time()
        
        # Simple spin loop with timeout
        while rclpy.ok() and not node.position_found:
            rclpy.spin_once(node, timeout_sec=0.5)
            
            if time.time() - start_time > args.timeout:
                print(f"ERROR: Could not get robot position within {args.timeout} seconds.")
                print("\n🔧 Available position topics:")
                topics = node.get_topic_names_and_types()
                for topic, _ in topics:
                    if any(keyword in topic.lower() for keyword in ['odom', 'pose', 'position']):
                        print(f"  {topic}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()