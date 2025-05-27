#!/usr/bin/env python3
# filepath: /home/ragib/clearpath_ws_2/src/clearpath_nav2_demos/gps/gps_waypoint_sequence.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import sys

class GPSWaypointSequence(Node):
    def __init__(self, waypoints):
        super().__init__('gps_waypoint_sequence')
        
        # Store waypoints
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.goal_reached = True
        self.origin_received = False
        
        # Publishers
        self.target_pub = self.create_publisher(NavSatFix, '/target_gps_coord', 10)
        
        # Subscribe to auto-set origin and waypoint status
        self.auto_origin_sub = self.create_subscription(
            NavSatFix, '/auto_map_origin', self.auto_origin_callback, 10)
        self.status_sub = self.create_subscription(
            NavSatFix, '/waypoint_status', self.waypoint_status_callback, 10)
        
        # Timer for waypoint publishing (starts after origin is received)
        self.waypoint_timer = None
        
        self.get_logger().info(f"🚀 GPS Waypoint Sequence loaded with {len(waypoints)} waypoints")
        self.get_logger().info("⏳ Waiting for automatic map origin from robot GPS...")
        self.print_waypoint_plan()
    
    def print_waypoint_plan(self):
        """Print the planned waypoint sequence"""
        self.get_logger().info("📍 Planned waypoint sequence:")
        for i, (lat, lon) in enumerate(self.waypoints):
            self.get_logger().info(f"   Waypoint {i+1}: {lat:.8f}, {lon:.8f}")
    
    def auto_origin_callback(self, msg):
        """Receive auto-set map origin from GPS navigation node"""
        if not self.origin_received:
            self.get_logger().info(f"✅ Received auto-set origin: {msg.latitude:.8f}, {msg.longitude:.8f}")
            self.get_logger().info("🎯 Starting waypoint sequence in 3 seconds...")
            
            self.origin_received = True
            
            # Start waypoint sequence after origin is set
            self.waypoint_timer = self.create_timer(3.0, self.publish_current_waypoint)
    
    def publish_current_waypoint(self):
        """Publish current waypoint when ready"""
        if self.goal_reached and self.current_waypoint < len(self.waypoints):
            lat, lon = self.waypoints[self.current_waypoint]
            
            msg = NavSatFix()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "gps"
            msg.latitude = lat
            msg.longitude = lon
            msg.altitude = 146.0  # Default altitude
            
            self.target_pub.publish(msg)
            self.get_logger().info(f"🎯 Sent waypoint {self.current_waypoint + 1}/{len(self.waypoints)}: {lat:.8f}, {lon:.8f}")
            
            # Wait for completion
            self.goal_reached = False
            
        elif self.current_waypoint >= len(self.waypoints):
            self.get_logger().info("🎉 All waypoints completed successfully!")
            if self.waypoint_timer:
                self.waypoint_timer.cancel()
            rclpy.shutdown()
    
    def waypoint_status_callback(self, msg):
        """Handle waypoint status updates"""
        if msg.status.status == 1:  # Waypoint reached
            self.get_logger().info(f"✅ Waypoint {self.current_waypoint + 1} reached!")
            self.current_waypoint += 1
            self.goal_reached = True
            
        elif msg.status.status == 2:  # Navigation failed
            self.get_logger().error(f"❌ Waypoint {self.current_waypoint + 1} failed! Moving to next...")
            self.current_waypoint += 1
            self.goal_reached = True

def main():
    # Define your target GPS coordinates here
    # The robot's current position will automatically be set as map origin
    waypoints = [
        # GPS_1: Your recorded position 1
        (41.87179475050352, -87.64911466370145),
        
        # GPS_2: Your recorded position 2  
        (41.87178371175604, -87.64940238122001),
        
        # GPS_3: Add more waypoints as needed
        # (41.87175220163368, -87.64923976708063),  # Uncomment to add GPS_3
        
        # You can add as many waypoints as you want:
        # (your_lat_4, your_lon_4),
        # (your_lat_5, your_lon_5),
    ]
    
    # Allow command line waypoints
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("GPS Waypoint Sequence Usage:")
            print("  python3 gps_waypoint_sequence.py [lat1 lon1 lat2 lon2 ...]")
            print("")
            print("Examples:")
            print("  # Use predefined waypoints:")
            print("  python3 gps_waypoint_sequence.py")
            print("")
            print("  # Use custom waypoints:")
            print("  python3 gps_waypoint_sequence.py 41.87179475 -87.64911466 41.87178371 -87.64940238")
            print("")
            print("Note: Robot's current GPS position will automatically be set as map origin")
            return
            
        coords = [float(x) for x in sys.argv[1:]]
        if len(coords) % 2 == 0:
            waypoints = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            print(f"✅ Using {len(waypoints)} waypoints from command line")
        else:
            print("❌ Error: Waypoints must be in pairs (lat lon lat lon ...)")
            return
    
    rclpy.init()
    
    node = GPSWaypointSequence(waypoints)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Navigation cancelled by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()