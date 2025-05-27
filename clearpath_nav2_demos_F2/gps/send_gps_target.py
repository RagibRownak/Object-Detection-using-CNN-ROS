#!/usr/bin/env python3
# filepath: /home/ragib/clearpath_ws_2/src/clearpath_nav2_demos/gps/send_gps_target.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import sys

class GPSTargetSender(Node):
    def __init__(self, lat, lon):
        super().__init__('gps_target_sender')
        
        self.target_pub = self.create_publisher(NavSatFix, '/target_gps_coord', 10)
        
        # Wait for publisher to be ready
        self.timer = self.create_timer(1.0, lambda: self.send_target(lat, lon))
        self.sent = False
    
    def send_target(self, lat, lon):
        if not self.sent:
            msg = NavSatFix()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "gps"
            msg.latitude = lat
            msg.longitude = lon
            msg.altitude = 146.0
            
            self.target_pub.publish(msg)
            self.get_logger().info(f"🎯 Sent GPS target: {lat:.8f}, {lon:.8f}")
            self.sent = True
            self.timer.cancel()

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 send_gps_target.py <latitude> <longitude>")
        print("Example: python3 send_gps_target.py 41.87179475 -87.64911466")
        return
    
    lat = float(sys.argv[1])
    lon = float(sys.argv[2])
    
    rclpy.init()
    node = GPSTargetSender(lat, lon)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()