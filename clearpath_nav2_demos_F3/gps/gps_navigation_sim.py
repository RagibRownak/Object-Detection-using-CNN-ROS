#!/usr/bin/env python3
# filepath: /home/ragib/clearpath_ws_2/src/clearpath_nav2_demos/gps/gps_navigation_sim.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import utm
from std_msgs.msg import Float64MultiArray, Int8
import os

class GPSNavigation(Node):
    def __init__(self):
        super().__init__('gps_navigation_node')
        
        # Fixed namespace for simulation
        namespace = "a200_1093"
        self.get_logger().info(f"Using fixed namespace: {namespace}")
        
        # Create Action Client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, f'/{namespace}/navigate_to_pose')
        
        # Subscribers
        self.map_origin_sub = self.create_subscription(NavSatFix, '/map_origin_gps', self.map_origin_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/target_gps_coord', self.gps_callback, 10)
        
        # UTM origin reference (Set from a separate topic)
        self.utm_origin_easting = None
        self.utm_origin_northing = None
        self.utm_zone_number = None  # Zone will be set dynamically
        
        # Goal status tracking
        self.goal_active = False  # Ensures we only process one goal at a time
    
    def map_origin_callback(self, msg):
        """ Store the fixed map origin GPS coordinates from a separate topic. """
        lat, lon = msg.latitude, msg.longitude
        utm_x, utm_y, zone_number, _ = utm.from_latlon(lat, lon)
        
        self.utm_origin_easting = utm_x
        self.utm_origin_northing = utm_y
        self.utm_zone_number = zone_number
        
        self.get_logger().info(f"Set Map Origin: Lat={lat}, Lon={lon}, UTM X={utm_x}, UTM Y={utm_y}")
    
    def gps_callback(self, msg):
        """ Convert target GPS to UTM and then to Map Frame """
        if self.utm_origin_easting is None:
            self.get_logger().warn("Map origin GPS not received yet! Ignoring target GPS.")
            return
        
        if self.goal_active:
            self.get_logger().info("Goal is still active. Ignoring new GPS input.")
            return  # Ignore new GPS data until the current goal completes
        
        lat, lon = msg.latitude, msg.longitude
        self.get_logger().info(f"Received Target GPS: Lat={lat}, Lon={lon}")
        
        # Convert Target GPS to UTM
        utm_x, utm_y, zone_number, _ = utm.from_latlon(lat, lon)
        
        # Convert UTM to `map` frame by subtracting the fixed UTM origin
        map_x = utm_x - self.utm_origin_easting
        map_y = utm_y - self.utm_origin_northing
        
        self.get_logger().info(f"Converted to Map Frame: X={map_x}, Y={map_y}")
        
        # Send to Nav2
        self.send_navigation_goal(map_x, map_y)
    
    def send_navigation_goal(self, x, y):
        """ Send a navigation goal to Nav2 action server """
        self.get_logger().info(f"Sending Goal to Nav2: X={x}, Y={y}")
        
        # Set goal as active
        self.goal_active = True
        
        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available! Navigation goal aborted.")
            self.goal_active = False
            return
        
        # Create the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0  # Identity quaternion
        
        # Create and send the goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        # Send the goal
        self.nav_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.goal_feedback_callback
        ).add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.goal_active = False
            return
        
        self.get_logger().info('Goal accepted')
        
        # Get the result
        future_result = goal_handle.get_result_async()
        future_result.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Goal finished with result: {result}')
        self.goal_active = False
    
    def goal_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Just log feedback if needed

def main(args=None):
    rclpy.init(args=args)
    gps_nav = GPSNavigation()
    
    try:
        rclpy.spin(gps_nav)
    except KeyboardInterrupt:
        gps_nav.get_logger().info("Terminating GPS Navigation Node...")
    finally:
        gps_nav.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()