#!/usr/bin/env python3
# filepath: /home/ragib/clearpath_ws_2/src/clearpath_nav2_demos/gps/gps_navigation.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import utm
from std_msgs.msg import Float64MultiArray, Int8
import yaml
import os

def load_namespace():
    yaml_path = "/home/administrator/colcon_ws/clearpath/robot.yaml"
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            return config['system']['ros2']['namespace']
    except Exception as e:
        print(f"Error loading namespace from YAML: {e}")
        return "a200_1093"

class GPSNavigation(Node):
    def __init__(self):
        super().__init__('gps_navigation_node')

        namespace = load_namespace()
        self.get_logger().info(f"Using namespace: {namespace}")
        
        # Create Action Client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, f'{namespace}/navigate_to_pose')

        # Subscribe to robot's actual GPS for automatic origin setting
        self.robot_gps_sub = self.create_subscription(
            NavSatFix, 
            f'/{namespace}/sensors/gps_0/navsatfix', 
            self.robot_gps_callback, 
            10
        )

        # Subscribers for manual commands
        self.map_origin_sub = self.create_subscription(NavSatFix, '/map_origin_gps', self.map_origin_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/target_gps_coord', self.gps_callback, 10)

        # Publishers
        self.status_pub = self.create_publisher(NavSatFix, '/waypoint_status', 10)
        self.auto_origin_pub = self.create_publisher(NavSatFix, '/auto_map_origin', 10)

        # UTM origin reference
        self.utm_origin_easting = None
        self.utm_origin_northing = None
        self.utm_zone_number = None
        self.utm_zone_letter = None
        
        # Auto-origin setup
        self.auto_origin_set = False
        self.origin_gps_lat = None
        self.origin_gps_lon = None

        # Goal status tracking
        self.goal_active = False

        self.get_logger().info("🚀 GPS Navigation started - waiting for robot GPS to set map origin...")

    def robot_gps_callback(self, msg):
        """Automatically set map origin from robot's current GPS position"""
        # Only set origin once and if we have a valid GPS fix
        if (not self.auto_origin_set and 
            msg.latitude != 0.0 and msg.longitude != 0.0 and 
            msg.status.status >= 0):  # Valid GPS fix
            
            # Set this GPS position as map origin
            self.origin_gps_lat = msg.latitude
            self.origin_gps_lon = msg.longitude
            
            # Convert to UTM for coordinate system
            utm_x, utm_y, zone_number, zone_letter = utm.from_latlon(
                self.origin_gps_lat, self.origin_gps_lon)

            self.utm_origin_easting = utm_x
            self.utm_origin_northing = utm_y
            self.utm_zone_number = zone_number
            self.utm_zone_letter = zone_letter
            
            self.auto_origin_set = True
            
            # Publish the auto-set origin for other nodes
            origin_msg = NavSatFix()
            origin_msg.header.stamp = self.get_clock().now().to_msg()
            origin_msg.header.frame_id = "gps"
            origin_msg.latitude = self.origin_gps_lat
            origin_msg.longitude = self.origin_gps_lon
            origin_msg.altitude = msg.altitude
            self.auto_origin_pub.publish(origin_msg)

            self.get_logger().info("📍 AUTO-SET Map Origin from robot GPS:")
            self.get_logger().info(f"   GPS: {self.origin_gps_lat:.8f}, {self.origin_gps_lon:.8f}")
            self.get_logger().info(f"   UTM: X={utm_x:.2f}, Y={utm_y:.2f}, Zone={zone_number}{zone_letter}")
            self.get_logger().info("✅ Ready to receive GPS waypoints!")

    def map_origin_callback(self, msg):
        """Manual map origin override (optional)"""
        if self.auto_origin_set:
            self.get_logger().warn("Map origin already auto-set from robot GPS. Ignoring manual override.")
            return
            
        lat, lon = msg.latitude, msg.longitude
        utm_x, utm_y, zone_number, zone_letter = utm.from_latlon(lat, lon)

        self.utm_origin_easting = utm_x
        self.utm_origin_northing = utm_y
        self.utm_zone_number = zone_number
        self.utm_zone_letter = zone_letter
        self.auto_origin_set = True

        self.get_logger().info(f"📍 MANUAL Map Origin: Lat={lat:.8f}, Lon={lon:.8f}")

    def gps_callback(self, msg):
        """Convert target GPS to UTM and send to Nav2"""
        if not self.auto_origin_set:
            self.get_logger().warn("Map origin not set yet! Waiting for robot GPS...")
            return

        if self.goal_active:
            self.get_logger().info("Goal is still active. Ignoring new GPS input.")
            return

        lat, lon = msg.latitude, msg.longitude
        self.get_logger().info(f"🎯 Received Target GPS: {lat:.8f}, {lon:.8f}")

        # Convert Target GPS to UTM
        utm_x, utm_y, zone_number, zone_letter = utm.from_latlon(lat, lon)

        # Convert to map frame coordinates
        map_x = utm_x - self.utm_origin_easting
        map_y = utm_y - self.utm_origin_northing

        # Calculate distance from origin
        distance = (map_x**2 + map_y**2)**0.5

        self.get_logger().info(f"📐 Map coordinates: X={map_x:.2f}m, Y={map_y:.2f}m")
        self.get_logger().info(f"📏 Distance from origin: {distance:.1f}m")

        # Send goal to Nav2
        self.send_navigation_goal(map_x, map_y)

    def send_navigation_goal(self, x, y):
        """Send goal to Nav2 in map frame"""
        if self.goal_active:
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"🚀 Sending goal to Nav2: X={x:.2f}m, Y={y:.2f}m")

        self.goal_active = True

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("❌ Nav2 action server not available!")
            self.goal_active = False
            return

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal response from Nav2"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ Goal Rejected!")
            self.goal_active = False
            return

        self.get_logger().info("✅ Goal Accepted! Navigating...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Handle navigation result"""
        result = future.result()
        status_msg = NavSatFix()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        
        if result.result:
            self.get_logger().info("🎉 Navigation Goal Reached!")
            status_msg.status.status = 1
        else:
            self.get_logger().error("❌ Navigation Failed!")
            status_msg.status.status = 2
        
        self.status_pub.publish(status_msg)
        self.goal_active = False

def main(args=None):
    rclpy.init(args=args)
    gps_navigation = GPSNavigation()
    
    try:
        rclpy.spin(gps_navigation)
    except KeyboardInterrupt:
        gps_navigation.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        if rclpy.ok():
            gps_navigation.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()

