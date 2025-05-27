#!/bin/bash
# filepath: /home/ragib/clearpath_ws_2/src/clearpath_nav2_demos/gps/sim_gps_waypoints.sh

# Start GPS navigation node first
echo "Starting GPS navigation node..."
ros2 run clearpath_nav2_demos gps_navigation.py --ros-args -p namespace:=a200_1093 &
GPS_NAV_PID=$!

# Wait for the node to initialize
echo "Waiting for GPS navigation node to initialize..."
sleep 5

# Set map origin (this will be treated as 0,0 in the map)
MAP_ORIGIN_LAT=43.5000
MAP_ORIGIN_LON=-80.5000

echo "Setting map origin to: $MAP_ORIGIN_LAT, $MAP_ORIGIN_LON"
# Use -1 instead of --once to avoid waiting for subscribers
ros2 topic pub /map_origin_gps sensor_msgs/msg/NavSatFix \
  "{header: {frame_id: 'map'}, latitude: $MAP_ORIGIN_LAT, longitude: $MAP_ORIGIN_LON, altitude: 0.0}" -1

# Wait for systems to process the origin
sleep 3

# Warehouse path waypoints
WAYPOINTS=(
  "43.5001 -80.5001"  # 10m north, 10m east
  "43.5003 -80.5001"  # 30m north, 10m east
  "43.5003 -80.5004"  # 30m north, 40m east
  "43.5001 -80.5004"  # 10m north, 40m east
  "43.5001 -80.5001"  # Back to start
)

# Navigate to each waypoint
for waypoint in "${WAYPOINTS[@]}"; do
  read lat lon <<< $waypoint
  echo "Navigating to waypoint: $lat, $lon"
  
  # Send waypoint
  ros2 topic pub /target_gps_coord sensor_msgs/msg/NavSatFix \
    "{header: {frame_id: 'map'}, latitude: $lat, longitude: $lon, altitude: 0.0}" -1
  
  # Wait for waypoint to be reached
  echo "Waiting for waypoint to be reached..."
  timeout=60  # Maximum wait time in seconds
  start_time=$(date +%s)

  while true; do
    # Check if robot is still moving (simplified approach)
    curr_time=$(date +%s)
    elapsed=$((curr_time - start_time))
    
    if [ $elapsed -gt $timeout ]; then
      echo "Timeout reached, moving to next waypoint"
      break
    fi
    
    # Try to check if goal is still active through a topic
    goal_active=$(ros2 topic echo --once /a200_1093/navigate_to_pose/_action/status 2>/dev/null | grep "status:" || echo "unknown")
    
    if [[ $goal_active == *"status: 4"* ]]; then
      echo "Goal succeeded, moving to next waypoint"
      break
    elif [[ $goal_active == *"status: 5"* ]]; then
      echo "Goal failed, moving to next waypoint anyway"
      break
    fi
    
    # Brief pause before checking again
    sleep 5
    echo "Still navigating... ($elapsed seconds elapsed)"
  done
  
  # Small pause between waypoints
  sleep 2
done

echo "All waypoints completed!"

# Clean up
if [ -n "$GPS_NAV_PID" ]; then
  echo "Stopping GPS navigation node..."
  kill $GPS_NAV_PID
fi