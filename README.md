# Object-Detection-using-CNN-ROS
This repository integrates Detectron2's panoptic segmentation with ROS 2 for intelligent environment understanding and autonomous navigation. The system processes RGB-D camera and LiDAR sensor data to create semantically segmented representations of the environment, enabling robots to identify objects, terrain, and obstacles in real-time.
ROS-Detectron2 Autonomous Navigation
A robust implementation of CNN-based panoptic segmentation for autonomous robot navigation, combining Detectron2 and ROS 2.
🔍 Overview
This project integrates state-of-the-art computer vision with ROS 2 for semantic understanding of environments and autonomous navigation. The system uses Detectron2's panoptic segmentation models to identify and classify objects in the environment, enabling safe and efficient path planning for mobile robots.
✨ Key Features

Real-time Panoptic Segmentation: Process camera feeds using Detectron2 models to identify objects, terrain types, and obstacles
ROS 2 Integration: Full integration with the ROS 2 ecosystem for modular robotics development
3D Point Cloud Segmentation: Colorize point cloud data based on semantic understanding
SLAM Integration: Compatible with modern SLAM systems for mapless navigation
Multi-Robot Support: Designed to support collaborative robot systems
Optimized Performance: Multi-threading and frame skipping options to balance between accuracy and processing speed

🛠️ Technical Details

Frameworks: ROS 2 (Humble), Detectron2, PyTorch
Camera Support: Intel RealSense, standard webcams, ROS image topics
Sensors: RGB cameras, depth sensors, LiDAR (via point cloud integration)
CNN Architectures: Panoptic FPN, Panoptic DeepLab
Hardware Acceleration: CUDA support for GPU acceleration

📊 Performance

Real-time segmentation at ~5-10 FPS on mid-range GPUs
Dynamic frame skipping to maintain system responsiveness
Processing thread pool for optimized CPU utilization
Configurable visualization options for debugging and development

🚀 Getting Started
Prerequisites
bash# ROS 2 Humble
sudo apt install ros-humble-desktop

# PyTorch and Detectron2
pip install torch torchvision
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# RealSense SDK (optional)
sudo apt install librealsense2-dev
pip install pyrealsense2
Installation
bash# Create a ROS workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone this repository
git clone https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

# Build the workspace
cd ~/ros2_ws
colcon build --symlink-install
Running the Demo
bash# Source ROS environment
source ~/ros2_ws/install/setup.bash

# Launch with webcam
ros2 run ros_detectron2_nav demo_ros.py --weights /path/to/model.pth --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml --webcam

# Launch with RealSense camera
ros2 run ros_detectron2_nav demo_ros.py --weights /path/to/model.pth --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml --webcam --realsense
🔄 ROS Topics
TopicTypeDescription/a200_1093/sensors/camera_0/color/image_panopticsensor_msgs/ImageRGB image with segmentation visualization/a200_1093/sensors/camera_0/depth/image_panopticsensor_msgs/ImageDepth image with segmentation overlay/a200_1093/sensors/camera_0/points_panopticsensor_msgs/PointCloud2Segmented point cloud with semantic coloring
🔧 Configuration
The system can be configured through command-line arguments:
--config-file: Path to Detectron2 model configuration file
--weights: Path to pre-trained model weights
--confidence-threshold: Minimum confidence score for detections (default: 0.5)
--skip-frames: Process 1 out of N+1 frames for performance (default: 0)
--num-threads: Number of processing threads (default: 1)
--direct-display: Show segmentation without blending with camera image
📚 References

Detectron2
ROS 2 Documentation
Panoptic Segmentation
Panoptic DeepLab

📝 Future Work

 Integration with Nav2 for autonomous navigation
 Real-time obstacle avoidance using segmentation data
 Training custom models for specific environments
 Support for more sensor types and fusion approaches
 Performance optimizations for edge computing devices
