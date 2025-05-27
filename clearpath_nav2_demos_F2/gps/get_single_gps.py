#!/usr/bin/env python3
# filepath: /home/ragib/clearpath_ws_2/src/clearpath_nav2_demos/gps/get_single_gps.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from tf2_ros import TransformListener, Buffer
import utm
import math
import sys
import time

class SingleGPSConverter(Node):
    def __init__(self, map_origin_lat=None, map_origin_lon=None):
        super().__init__('single_gps_converter')
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Robot namespace
        self.robot_ns = "a200_1093"
        
        # Flag to ensure we only get one reading
        self.position_found = False
        
        # GPS origin (will be set from robot's actual GPS if not provided)
        self.map_origin_lat = map_origin_lat
        self.map_origin_lon = map_origin_lon
        self.utm_origin_set = False
        
        # Subscribe to robot's actual GPS for origin (REAL ROBOT TOPIC)
        self.gps_sub = self.create_subscription(
            NavSatFix, 
            f'/{self.robot_ns}/sensors/gps_0/navsatfix',  # CORRECT TOPIC
            self.gps_callback, 
            10
        )
        
        # Subscribe to GPS-based odometry (direct GPS position)
        self.gps_odom_sub = self.create_subscription(
            Odometry, 
            f'/{self.robot_ns}/sensors/gps_0/odom',
            lambda msg: self.odom_callback(msg, "GPS Odom"), 
            10
        )
        
        # Subscribe to platform odometry sources
        self.odom_subs = [
            self.create_subscription(Odometry, f'/{self.robot_ns}/platform/odom', 
                                   lambda msg: self.odom_callback(msg, "Platform Odom"), 10),
            self.create_subscription(Odometry, f'/{self.robot_ns}/platform/odom/filtered', 
                                   lambda msg: self.odom_callback(msg, "Filtered Odom"), 10),
        ]
        
        # Subscribe to AMCL pose as backup
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f'/{self.robot_ns}/amcl_pose', 
            self.amcl_callback,
            10
        )
        
        # Timer to get position via TF2 (should work on real robot)
        self.timer = self.create_timer(2.0, self.get_position_from_tf)
        
        self.get_logger().info("Getting robot position and converting to GPS...")
        self.get_logger().info("Will use robot's actual GPS as map origin")
        
        # Check environment type
        self.check_robot_type()
        
    def check_robot_type(self):
        """Detect if running on real robot or simulation"""
        topics = self.get_topic_names_and_types()
        topic_names = [topic for topic, _ in topics]
        
        if '/clock' in topic_names:
            self.get_logger().info("🤖 Detected: Simulation environment")
        else:
            self.get_logger().info("🔧 Detected: Real robot environment")
        
    def gps_callback(self, msg):
        """Get GPS coordinate and set as map origin if needed"""
        # Check if GPS has valid fix
        if (msg.latitude != 0.0 and msg.longitude != 0.0 and 
            msg.status.status >= 0):  # GPS status OK
            
            # If no manual origin set, use robot's current GPS as origin
            if self.map_origin_lat is None or self.map_origin_lon is None:
                self.map_origin_lat = msg.latitude
                self.map_origin_lon = msg.longitude
                self.setup_utm_origin()
                self.get_logger().info(f"Set map origin from robot GPS: {self.map_origin_lat:.8f}, {self.map_origin_lon:.8f}")
            
            # If we have a position in map coordinates, this callback handles direct GPS output
            if not self.position_found:
                self.print_gps_results(msg.latitude, msg.longitude, "Direct GPS", "WGS84")
                self.position_found = True
                self.shutdown_node()
    
    def setup_utm_origin(self):
        """Setup UTM conversion from map origin"""
        if self.map_origin_lat and self.map_origin_lon:
            self.utm_origin_easting, self.utm_origin_northing, self.utm_zone_number, self.utm_zone_letter = utm.from_latlon(
                self.map_origin_lat, self.map_origin_lon
            )
            self.utm_origin_set = True
        
    def odom_callback(self, msg, source):
        """Get position from odometry and convert to GPS"""
        if self.position_found:
            return
        
        # Wait for GPS origin to be set
        if not self.utm_origin_set:
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Convert to GPS
        gps_lat, gps_lon = self.map_to_gps(x, y)
        
        # Print results
        self.print_conversion_results(x, y, gps_lat, gps_lon, source, msg.header.frame_id)
        
        self.position_found = True
        self.shutdown_node()
        
    def amcl_callback(self, msg):
        """Get position from AMCL pose"""
        if self.position_found:
            return
        
        # Wait for GPS origin to be set
        if not self.utm_origin_set:
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Convert to GPS
        gps_lat, gps_lon = self.map_to_gps(x, y)
        
        # Print results
        self.print_conversion_results(x, y, gps_lat, gps_lon, "AMCL", msg.header.frame_id)
        
        self.position_found = True
        self.shutdown_node()
    
    def get_position_from_tf(self):
        """Get position using TF2 (should work on real robot)"""
        if self.position_found:
            return
        
        # Wait for GPS origin to be set
        if not self.utm_origin_set:
            return
            
        # Try different frame combinations
        frame_combinations = [
            ('map', 'base_link'),
            (f'{self.robot_ns}/map', f'{self.robot_ns}/base_link'),
            ('map', f'{self.robot_ns}/base_link'),
            (f'{self.robot_ns}/odom', f'{self.robot_ns}/base_link'),
            ('odom', 'base_link'),
        ]
        
        for target_frame, source_frame in frame_combinations:
            try:
                # Get transform
                transform = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time())
                
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                
                # Convert to GPS
                gps_lat, gps_lon = self.map_to_gps(x, y)
                
                # Print results
                self.print_conversion_results(x, y, gps_lat, gps_lon, 
                                            f"TF2: {target_frame}->{source_frame}", target_frame)
                
                self.position_found = True
                self.shutdown_node()
                return
                
            except Exception as e:
                continue
        
        self.get_logger().debug("Still waiting for valid transform...")
    
    def map_to_gps(self, map_x, map_y):
        """Convert map coordinates to GPS coordinates"""
        # Add map coordinates to UTM origin
        utm_easting = self.utm_origin_easting + map_x
        utm_northing = self.utm_origin_northing + map_y
        
        # Convert back to GPS
        lat, lon = utm.to_latlon(utm_easting, utm_northing, 
                                self.utm_zone_number, self.utm_zone_letter)
        
        return lat, lon
    
    def print_gps_results(self, lat, lon, source, frame_id):
        """Print direct GPS results"""
        print(f"\n=== Robot GPS Coordinate ===")
        print(f"Source: {source}")
        print(f"Reference Frame: {frame_id}")
        print(f"GPS Coordinate: {lat:.8f}, {lon:.8f}")
        print(f"Google Maps: https://www.google.com/maps?q={lat},{lon}")
        print(f"OpenStreetMap: https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18")
        print("============================\n")
    
    def print_conversion_results(self, x, y, gps_lat, gps_lon, source, frame_id=""):
        """Print the conversion results"""
        print(f"\n=== Robot Position to GPS Conversion ===")
        print(f"Source: {source}")
        if frame_id:
            print(f"Reference Frame: {frame_id}")
        print(f"Map Origin GPS: {self.map_origin_lat:.8f}, {self.map_origin_lon:.8f}")
        print(f"Robot Map Position: x={x:.3f}m, y={y:.3f}m")
        print(f"Robot GPS Coordinate: {gps_lat:.8f}, {gps_lon:.8f}")
        print(f"Google Maps: https://www.google.com/maps?q={gps_lat},{gps_lon}")
        print(f"OpenStreetMap: https://www.openstreetmap.org/?mlat={gps_lat}&mlon={gps_lon}&zoom=18")
        print("=========================================\n")
    
    def shutdown_node(self):
        """Shutdown the node properly"""
        if hasattr(self, 'timer'):
            self.timer.cancel()
        rclpy.shutdown()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Get GPS coordinate from robot position')
    parser.add_argument('--origin-lat', type=float, default=None, 
                       help='Map origin latitude (default: use robot GPS)')
    parser.add_argument('--origin-lon', type=float, default=None,
                       help='Map origin longitude (default: use robot GPS)')
    parser.add_argument('--timeout', type=float, default=15.0,
                       help='Timeout in seconds (default: 15.0)')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    node = SingleGPSConverter(
        map_origin_lat=args.origin_lat,
        map_origin_lon=args.origin_lon
    )
    
    try:
        start_time = time.time()
        
        # Simple spin loop with timeout
        while rclpy.ok() and not node.position_found:
            rclpy.spin_once(node, timeout_sec=0.5)
            
            if time.time() - start_time > args.timeout:
                print(f"ERROR: Could not get robot position within {args.timeout} seconds.")
                print("\n🔧 Available topics:")
                topics = node.get_topic_names_and_types()
                for topic, _ in topics:
                    if any(keyword in topic.lower() for keyword in ['odom', 'pose', 'position', 'gps', 'navsatfix']):
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