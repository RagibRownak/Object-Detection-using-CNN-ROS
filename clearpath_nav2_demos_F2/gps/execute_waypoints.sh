#!/bin/bash
# filepath: /home/ragib/clearpath_ws_2/src/clearpath_nav2_demos/gps/execute_waypoints.sh

echo "🚀 Starting Automatic GPS Waypoint Navigation"
echo ""
echo "📍 How it works:"
echo "   1. Robot's current GPS position becomes map origin (0,0)"
echo "   2. Navigate to specified GPS waypoints"
echo "   3. All coordinates are relative to starting position"
echo ""

# Check if GPS navigation is already running
if ! ros2 node list | grep -q "gps_navigation_node"; then
  echo "🔧 Starting GPS navigation node..."
  python3 gps_navigation.py &
  GPS_NAV_PID=$!
  sleep 3
  echo "✅ GPS navigation node started"
else
  echo "✅ GPS navigation node already running"
  GPS_NAV_PID=""
fi

# Start waypoint sequence
echo ""
echo "🎯 Starting waypoint sequence..."
echo "⏳ Waiting for robot GPS to set map origin..."
python3 gps_waypoint_sequence.py

echo ""
echo "🎉 GPS Navigation sequence completed!"

# Clean up
if [ -n "$GPS_NAV_PID" ]; then
  echo "🔧 Stopping GPS navigation node..."
  kill $GPS_NAV_PID
  echo "✅ Cleanup complete"
fi