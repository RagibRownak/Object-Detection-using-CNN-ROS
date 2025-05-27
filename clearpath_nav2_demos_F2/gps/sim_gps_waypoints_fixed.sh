#!/bin/bash

# Ensure the a200_1093 simulation is running first with:
# ros2 launch clearpath_nav2_demos a200_1093_demo.launch.py

# Start GPS navigation node with fixed namespace
echo "Starting GPS navigation node..."
ros2 run clearpath_nav2_demos gps_navigation_sim.py &
GPS_NAV_PID=$!

# Wait for the node to initialize
echo "Waiting for GPS navigation node to initialize..."
sleep 10

# Set map origin (this will be treated as 0,0 in the map)
MAP_ORIGIN_LAT=43.5000
MAP_ORIGIN_LON=-80.5000

echo "Setting map origin to: $MAP_ORIGIN_LAT, $MAP_ORIGIN_LON"
ros2 topic pub -1 /map_origin_gps sensor_msgs/msg/NavSatFix \
  "{header: {frame_id: 'map'}, latitude: $MAP_ORIGIN_LAT, longitude: $MAP_ORIGIN_LON, altitude: 0.0}"

# Wait for systems to process the origin
sleep 5

# Warehouse path waypoints - now using 8-meter spacing
# At latitude 43.5N: 
# - 0.000072 degrees latitude ≈ 8 meters north/south
# - 0.000099 degrees longitude ≈ 8 meters east/west
WAYPOINTS=(
  "43.5000 -80.500099"  # Origin + 8m east
  "43.500072 -80.500099"  # 8m north, 8m east
  "43.500072 -80.5000"  # 8m north, back at original longitude
  "43.5000 -80.5000"  # Back to origin
)

echo "Starting waypoint navigation sequence..."
# Navigate to each waypoint with longer timeouts for simulation
for waypoint in "${WAYPOINTS[@]}"; do
  read lat lon <<< $waypoint
  echo "Navigating to waypoint: $lat, $lon"
  
  # Send waypoint
  ros2 topic pub -1 /target_gps_coord sensor_msgs/msg/NavSatFix \
    "{header: {frame_id: 'map'}, latitude: $lat, longitude: $lon, altitude: 0.0}"
  
  # Simplified wait with fixed timeout - simulation can be slow
  echo "Waiting for navigation (120 seconds max)..."
  sleep 120
  
  echo "Moving to next waypoint"
  sleep 5
done

echo "All waypoints completed!"

# Clean up
if [ -n "$GPS_NAV_PID" ]; then
  echo "Stopping GPS navigation node..."
  kill $GPS_NAV_PID
fi