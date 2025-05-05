#!/usr/bin/env python3

import argparse
import numpy as np
import os
import time
import warnings
import cv2
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import LaserScan
import math
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from cv_bridge import CvBridge
from std_msgs.msg import Header
import struct
import random

# Add warning filter
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from torch.amp import autocast

# Constants
WINDOW_NAME = "Detectron2 Panoptic Segmentation"


class PanopticSegmentationNode(Node):
    """ROS 2 Node for processing camera feeds and publishing segmentation results"""
    
    def __init__(self, config_file, weights, confidence_threshold=0.5, 
                 color_topic=None, depth_topic=None, points_topic=None, visualize=True):
        super().__init__('panoptic_segmentation_node')
        
        # Initialize parameters
        self.bridge = CvBridge()
        self.visualize = visualize
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_pointcloud = None
        self.processing = False
        self.color_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.points_lock = threading.Lock()
        self.latest_panoptic_result = None
        self.result_lock = threading.Lock()
        self.camera_info = None
        self.color_frame_id = "camera_color_optical_frame"
        self.depth_frame_id = "camera_depth_optical_frame"
        self.last_segmented_cloud = None  # Cache for the last segmented cloud
        self.last_segment_info = None     # Cache for the segment info
        self.segmentation_lock = threading.Lock()  # Lock for segmentation data
        
        # Add after other initialization in __init__
        self.vis_buffer = None  # Visualization buffer for double-buffering
        self.vis_lock = threading.Lock()  # Lock for visualization buffer
        self.drop_frames = False  # Flag to drop frames when processing can't keep up
        self.consecutive_errors = 0  # Count consecutive errors to detect problems
        self.max_consecutive_errors = 3  # Maximum allowed consecutive errors before recovery
        
        # Set up logger
        self.logger = setup_logger(name="panoptic_ros")
        self.logger.info("Initializing Panoptic Segmentation Node")
        
        # Set up Detectron2
        self.logger.info(f"Loading model from {weights} with config {config_file}")
        cfg = self._setup_cfg(config_file, weights, confidence_threshold)
        self.demo = VisualizationDemo(cfg)
        
        # Create publishers for segmentation results
        self.color_publisher = self.create_publisher(
            ROSImage, '/a200_1093/sensors/camera_0/color/image_panoptic', 10)
        self.get_logger().info("Publisher initialized on /a200_1093/sensors/camera_0/color/image_panoptic")
        
        self.depth_publisher = self.create_publisher(
            ROSImage, '/a200_1093/sensors/camera_0/depth/image_panoptic', 10)
        self.get_logger().info("Publisher initialized on /a200_1093/sensors/camera_0/depth/image_panoptic")
        
        self.points_publisher = self.create_publisher(
            PointCloud2, '/a200_1093/sensors/camera_0/points_panoptic', 10)
        self.get_logger().info("Publisher initialized on /a200_1093/sensors/camera_0/points_panoptic")
        
        # Create a publisher for laser scan data
        self.scan_publisher = self.create_publisher(
            LaserScan, '/scan', 10)  # Nav2 will use the /scan topic
        self.get_logger().info("Publisher initialized on /scan")

        # Create TF broadcaster for static transforms if needed
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transforms()
        
        # Set up subscribers
        if color_topic:
            self.color_sub = self.create_subscription(
                ROSImage,
                color_topic,
                self.color_callback,
                10)
            self.get_logger().info(f"Subscribed to color topic: {color_topic}")
            
        if depth_topic:
            self.depth_sub = self.create_subscription(
                ROSImage,
                depth_topic,
                self.depth_callback,
                10)
            self.get_logger().info(f"Subscribed to depth topic: {depth_topic}")
            
        if points_topic:
            self.points_sub = self.create_subscription(
                PointCloud2,
                points_topic,
                self.points_callback,
                10)
            self.get_logger().info(f"Subscribed to points topic: {points_topic}")
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/a200_1093/sensors/camera_0/color/camera_info',
            self.camera_info_callback,
            10)
        
        # Start processing thread
        self.running = True
        self.process_thread = threading.Thread(target=self.process_images)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start pointcloud processing thread if needed
        if points_topic:
            self.points_thread = threading.Thread(target=self.process_pointcloud)
            self.points_thread.daemon = True
            self.points_thread.start()
        
        # Add in __init__ after other initializations
        self.create_timer(0.2, self.publish_test_segmented_pointcloud)  # Run at 5Hz instead of 1Hz
        
    def _setup_cfg(self, config_file, weights_path, confidence_threshold):
        """Set up Detectron2 configuration"""
        cfg = get_cfg()
        # Add panoptic deeplab config
        from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
        add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.freeze()
        return cfg
        
    def publish_static_transforms(self):
        """Publish static transforms between frames"""
        # Transform from camera frame to laser scan frame
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = self.color_frame_id
        
        # Set transform values based on your camera mount position
        # Adjust these values according to your robot setup
        transform.transform.translation.x = 0.0  # Forward offset from base_link
        transform.transform.translation.y = 0.0  # Left offset from base_link
        transform.transform.translation.z = 0.2  # Height above base_link
        
        # Set rotation (if camera is forward-facing)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        # Publish the transform
        self.tf_broadcaster.sendTransform(transform)
        
        # Also publish transform from base_link to base_scan (for Nav2)
        scan_transform = TransformStamped()
        scan_transform.header.stamp = self.get_clock().now().to_msg()
        scan_transform.header.frame_id = "base_link"
        scan_transform.child_frame_id = "base_scan"
        
        # Usually the laser scan is at the same position as the camera
        scan_transform.transform.translation.x = 0.0
        scan_transform.transform.translation.y = 0.0
        scan_transform.transform.translation.z = 0.0
        
        # Identity rotation
        scan_transform.transform.rotation.x = 0.0
        scan_transform.transform.rotation.y = 0.0
        scan_transform.transform.rotation.z = 0.0
        scan_transform.transform.rotation.w = 1.0
        
        # Publish the transform
        self.tf_broadcaster.sendTransform(scan_transform)
        
        self.get_logger().info("Published static transforms")
    
    def color_callback(self, msg):
        """Process incoming color frame"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.color_lock:
                self.latest_color_image = cv_image
                self.color_frame_id = msg.header.frame_id
                self.get_logger().debug("Received color frame")
        except Exception as e:
            self.get_logger().error(f"Error processing color frame: {str(e)}")

    def depth_callback(self, msg):
        """Process incoming depth frame"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)  # Use default encoding for depth
            with self.depth_lock:
                self.latest_depth_image = cv_image
                self.depth_frame_id = msg.header.frame_id
                self.get_logger().debug("Received depth frame")
        except Exception as e:
            self.get_logger().error(f"Error processing depth frame: {str(e)}")
            
    def points_callback(self, msg):
        """Process incoming pointcloud data"""
        with self.points_lock:
            self.latest_pointcloud = msg
            self.get_logger().debug("Received pointcloud")
    
    def camera_info_callback(self, msg):
        """Store camera calibration parameters"""
        self.camera_info = msg
        self.get_logger().info("Received camera calibration")
    
    def process_images(self):
        """Process images in a separate thread"""
        while self.running and rclpy.ok():
            if self.processing:
                # If we're already processing and drop_frames is enabled, 
                # check if we should skip this frame
                if self.drop_frames:
                    with self.color_lock:
                        if self.latest_color_image is not None:
                            # Just update the display with the raw image to keep UI responsive
                            if self.visualize and self.vis_buffer is not None:
                                try:
                                    cv2.imshow(WINDOW_NAME, self.latest_color_image)
                                    cv2.waitKey(1)
                                except:
                                    pass
                time.sleep(0.01)
                continue
                
            with self.color_lock:
                color_img = self.latest_color_image
                if color_img is None:
                    time.sleep(0.01)
                    continue
                color_copy = color_img.copy()
            
            self.processing = True
            
            try:
                # Process with detectron2
                start_time = time.time()
                with autocast("cuda"):
                    predictions, visualized_output = self.demo.run_on_image(color_copy)

                # ====== DEPTH AUGMENTATION START ======

                # Extract depth information for each detected instance
                with self.depth_lock:
                    depth_img = self.latest_depth_image
                    depth_available = depth_img is not None

                if depth_available and "instances" in predictions:
                    instances = predictions["instances"].to("cpu")
                    if len(instances) > 0:
                        self.get_logger().info(f"Found {len(instances)} instances, extracting depth...")
                        
                        # Add depth information to predictions
                        depths = []
                        filtered_instances_idx = []
                        
                        # Correct way to access Instances data - iterate over indices, not the object itself
                        for idx in range(len(instances)):
                            if instances.has("pred_boxes"):
                                # Get the bounding box for this instance
                                box = instances.pred_boxes.tensor[idx].numpy()
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Clamp values to image size
                                x1 = max(0, min(x1, depth_img.shape[1]-1))
                                x2 = max(0, min(x2, depth_img.shape[1]-1))
                                y1 = max(0, min(y1, depth_img.shape[0]-1))
                                y2 = max(0, min(y2, depth_img.shape[0]-1))
                                
                                if x2 <= x1 or y2 <= y1:
                                    self.get_logger().warn(f"Invalid box for instance {idx}")
                                    continue
                                
                                # Crop depth region
                                depth_crop = depth_img[y1:y2, x1:x2]
                                
                                # Convert depth values if needed (unit conversion)
                                # Adjust these values based on your depth sensor
                                valid_depth = depth_crop[(depth_crop > 0.1) & (depth_crop < 10.0)]  # Only reasonable range (0.1m - 10m)
                                
                                if valid_depth.size > 0:
                                    median_depth = np.median(valid_depth)
                                    depths.append(median_depth)
                                    filtered_instances_idx.append(idx)
                                    
                                    # Get class name for the instance if available
                                    class_id = int(instances.pred_classes[idx].item()) if instances.has("pred_classes") else -1
                                    class_name = "unknown"
                                    try:
                                        metadata = MetadataCatalog.get(self.demo.cfg.DATASETS.TEST[0] 
                                                    if len(self.demo.cfg.DATASETS.TEST) else "__unused")
                                        if hasattr(metadata, "thing_classes") and class_id >= 0 and class_id < len(metadata.thing_classes):
                                            class_name = metadata.thing_classes[class_id]
                                    except:
                                        pass
                                    
                                    self.get_logger().info(f"Instance {idx} ({class_name}): median depth = {median_depth:.2f} meters")
                                else:
                                    self.get_logger().warn(f"No valid depth for instance {idx}")
                            else:
                                self.get_logger().warn(f"Instance {idx} has no pred_boxes field")
                                    
                        # Store depth information with predictions for later use
                        if depths:
                            # Add depth information as a new dictionary entry
                            predictions["depth_info"] = {
                                "instance_depths": depths,
                                "filtered_instances_idx": filtered_instances_idx
                            }
                    else:
                        self.get_logger().info("No instances found for depth extraction")

                # ====== DEPTH AUGMENTATION END ======

                process_time = time.time() - start_time
                self.get_logger().info(f"Processed frame in {process_time:.2f}s")

                # Log prediction keys and details
                prediction_keys = list(predictions.keys())
                self.get_logger().info(f"Prediction keys: {prediction_keys}")
                if "panoptic_seg" in predictions:
                    self.get_logger().info("Panoptic segmentation found!")
                if "instances" in predictions:
                    self.get_logger().info(f"Instances found: {len(predictions['instances'])}")

                # Store the latest panoptic segmentation result
                with self.result_lock:
                    self.latest_panoptic_result = {
                        'vis_output': visualized_output,
                        'predictions': predictions,
                        'raw_image': color_copy
                    }

                # Publish laser scan data
                self.publish_laser_scan(predictions)

                # Get the segmentation visualization
                vis_frame = visualized_output.get_image()

                # Create a BGR copy for OpenCV visualization
                vis_frame_bgr = vis_frame[:, :, ::-1]  # RGB to BGR

                # Draw bounding boxes
                vis_frame_with_boxes = self.draw_instance_boxes(vis_frame_bgr.copy(), predictions)

                # Publish to ROS
                self.publish_color_frame(vis_frame_with_boxes)
                
                # Process and publish depth frame if available
                with self.depth_lock:
                    depth_img = self.latest_depth_image
                    if depth_img is not None:
                        self.publish_depth_frame(depth_img, predictions)
                
                
                # Display if visualization is enabled - with double-buffering
                if self.visualize:
                    try:
                        # Use double-buffering for smoother visualization
                        if vis_frame_with_boxes is not None and vis_frame_with_boxes.size > 0:
                            with self.vis_lock:
                                # Store in buffer first
                                self.vis_buffer = vis_frame_with_boxes.copy()
                                
                                # Show from buffer
                                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                                cv2.imshow(WINDOW_NAME, self.vis_buffer)
                                key = cv2.waitKey(1)
                                if key == 27:  # Exit on ESC
                                    self.running = False
                                    break
                                elif key == 32:  # Space bar - toggle frame dropping
                                    self.drop_frames = not self.drop_frames
                                    self.get_logger().info(f"Frame dropping: {'ON' if self.drop_frames else 'OFF'}")
                        
                        # Reset error counter on successful visualization
                        self.consecutive_errors = 0
                    except Exception as e:
                        self.get_logger().warn(f"Visualization error: {str(e)}")
                        self.consecutive_errors += 1
                        
                        # If too many consecutive errors, try to recover
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            self.get_logger().warn("Too many visualization errors, attempting recovery...")
                            try:
                                # Destroy and recreate window
                                cv2.destroyWindow(WINDOW_NAME)
                                time.sleep(0.1)
                                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                                self.consecutive_errors = 0
                            except:
                                pass
                
            except Exception as e:
                # Add this special case handling:
                if "'dict' object has no attribute 'depth_info'" in str(e):
                    self.get_logger().warn("Ignoring depth_info attribute access error")
                    # Continue with remaining processing
                    pass
                else:
                    import traceback
                    self.get_logger().error(f"Error in processing: {str(e)}")
                    self.get_logger().error(traceback.format_exc())  # This prints the full stack trace
            finally:
                self.processing = False
            
            time.sleep(0.01)
        
        # Clean up on exit
        if self.visualize:
            cv2.destroyAllWindows()
    
    def process_pointcloud(self):
        """Process pointcloud data using the latest segmentation results"""
        while self.running and rclpy.ok():
            # Just sleep - we're handling everything in the timer-based publisher
            time.sleep(0.1)
            
            # Don't process here anymore - all handled by publish_test_segmented_pointcloud
            # This avoids conflicting with the timer-based publisher
    
    def apply_segmentation_to_pointcloud(self, pointcloud, segmentation_data):
        """Apply panoptic segmentation to the pointcloud"""
        try:
            # Get segmentation data correctly - this was the main issue!
            # The panoptic_seg is returned in a different format from detectron2
            if 'panoptic_seg' not in segmentation_data['predictions']:
                self.get_logger().error("No panoptic segmentation found in predictions")
                return pointcloud
                
            # Get data in the correct format
            panoptic_seg = segmentation_data['predictions']['panoptic_seg'][0].cpu().numpy()
            segments_info = segmentation_data['predictions']['panoptic_seg'][1]
            
            self.get_logger().info(f"Segmentation shape: {panoptic_seg.shape}, segments: {len(segments_info)}")
            
            # Print segment categories to verify segmentation
            segment_categories = [s.get('category_id', -1) for s in segments_info]
            self.get_logger().info(f"Segment categories: {segment_categories[:5]}...")
            
            # Extract point data as numpy array for processing
            points_xyz = self.pointcloud_to_numpy(pointcloud)
            if points_xyz is None:
                self.get_logger().error("Failed to extract points from pointcloud")
                return pointcloud  # Return original if extraction fails
            
            # Get camera parameters
            if self.camera_info is not None:
                K = self.camera_info.k
                fx = K[0]  # Focal length x
                fy = K[4]  # Focal length y
                cx = K[2]  # Principal point x
                cy = K[5]  # Principal point y
                width = self.camera_info.width
                height = self.camera_info.height
                self.get_logger().info(f"Using camera info: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            else:
                fx = 525.0  # default focal length x
                fy = 525.0  # default focal length y
                cx = 319.5  # default principal point x
                cy = 239.5  # default principal point y
                width = 640   # default width
                height = 480  # default height
            
            # Create a deep copy of the pointcloud
            new_pointcloud = PointCloud2(
                header=pointcloud.header,
                height=pointcloud.height,
                width=pointcloud.width,
                is_bigendian=pointcloud.is_bigendian,
                point_step=pointcloud.point_step,
                row_step=pointcloud.row_step,
                is_dense=pointcloud.is_dense,
                fields=pointcloud.fields,
                data=bytearray(pointcloud.data)
            )
            
            # Find rgb field in the pointcloud
            rgb_field = None
            rgb_offset = None
            for field in pointcloud.fields:
                self.get_logger().info(f"Pointcloud field: {field.name} at offset {field.offset}")
                if field.name == 'rgb' or field.name == 'rgba':
                    rgb_field = field
                    rgb_offset = field.offset
                    break
            
            # If no rgb field, we can't color the pointcloud
            if rgb_field is None:
                self.get_logger().error("No RGB field found in pointcloud, can't apply color segmentation")
                # Could create a new pointcloud with RGB fields here, but for simplicity we'll just return
                return pointcloud
            
            # Generate vibrant colors for each segment
            segment_colors = {}
            # Use a bright default color for unclassified points
            default_color = (200, 200, 200)  # Light gray
            
            for segment in segments_info:
                segment_id = segment['id']
                category_id = segment.get('category_id', 0)
                # Use category_id for consistent coloring
                random.seed(category_id)
                # Use more vibrant colors (high saturation)
                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                segment_colors[segment_id] = color
                random.seed()  # Reset seed
            
            # Reshape data for easier access
            new_data = np.frombuffer(new_pointcloud.data, dtype=np.uint8).reshape(-1, pointcloud.point_step)
            
            # Track statistics
            total_points = pointcloud.width * pointcloud.height
            colored_points = 0
            skipped_points = 0
            projection_failures = 0
            
            # Update colors in place
            for i, (x, y, z) in enumerate(points_xyz):
                # Skip invalid points
                if np.isnan(x) or np.isnan(y) or np.isnan(z) or z <= 0:
                    skipped_points += 1
                    continue
                    
                try:
                    # Project point to get segment color
                    u = int((x * fx) / z + cx)
                    v = int((y * fy) / z + cy)
                    
                    # If point projects into the image
                    if 0 <= u < width and 0 <= v < height:
                        # Get segment ID
                        segment_id = panoptic_seg[v, u]
                        
                        # Get color for this segment
                        if segment_id in segment_colors:
                            r, g, b = segment_colors[segment_id]
                        else:
                            # Use default color if segment not found
                            r, g, b = default_color
                            
                        # Pack RGB in PCL format expected by RViz
                        rgb_val = (r << 16) | (g << 8) | b
                        # Convert to float32 for PCL
                        rgb_float = struct.unpack('f', struct.pack('I', rgb_val))[0]
                        
                        # Pack as bytes
                        rgb_bytes = struct.pack('f', rgb_float)
                        
                        # Update the RGB value in-place
                        new_data[i, rgb_offset:rgb_offset+4] = np.frombuffer(rgb_bytes, dtype=np.uint8)
                        colored_points += 1
                    else:
                        projection_failures += 1
                except (ValueError, IndexError, OverflowError, ZeroDivisionError) as e:
                    # Skip problematic points
                    self.get_logger().debug(f"Error at point {i}: {e}")
                    continue
            
            # Log detailed statistics
            self.get_logger().info(f"Colored {colored_points}/{total_points} points with segment colors")
            self.get_logger().info(f"Skipped {skipped_points} invalid points")
            self.get_logger().info(f"Projection failures: {projection_failures}")
            
            # Create message with the updated data
            new_pointcloud.data = bytes(new_data.tobytes())
            
            # Log a message to remind about RViz settings
            self.get_logger().info("For RViz: Set Position Transformer to XYZ and Color Transformer to RGB8")
            
            return new_pointcloud
            
        except Exception as e:
            self.get_logger().error(f"Error in pointcloud segmentation: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return pointcloud  # Return original on error
    
    def pointcloud_to_numpy(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        try:
            # Get cloud data as array of bytes
            cloud_data = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
            
            # Find the offsets for x, y, z in the point cloud fields
            x_offset = None
            y_offset = None
            z_offset = None
            
            for field in cloud_msg.fields:
                if field.name == 'x':
                    x_offset = field.offset
                elif field.name == 'y':
                    y_offset = field.offset
                elif field.name == 'z':
                    z_offset = field.offset
            
            if x_offset is None or y_offset is None or z_offset is None:
                self.get_logger().error("Point cloud does not have x, y, z fields")
                return None
            
            # Extract x, y, z coordinates
            x = np.frombuffer(cloud_data[:, x_offset:x_offset+4].tobytes(), dtype=np.float32)
            y = np.frombuffer(cloud_data[:, y_offset:y_offset+4].tobytes(), dtype=np.float32)
            z = np.frombuffer(cloud_data[:, z_offset:z_offset+4].tobytes(), dtype=np.float32)
            
            # Debug - check if points appear valid
            valid_points = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            self.get_logger().debug(f"Valid points in cloud: {np.sum(valid_points)}/{len(x)}")
            
            return np.column_stack((x, y, z))
        except Exception as e:
            self.get_logger().error(f"Failed to convert pointcloud: {str(e)}")
            return None
    
    def publish_color_frame(self, cv_image):
        """Convert OpenCV image to ROS message and publish color frame"""
        try:
            # Convert BGR to RGB for ROS
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            ros_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = self.color_frame_id
            self.color_publisher.publish(ros_image)
            self.get_logger().debug("Published color segmentation result")
        except Exception as e:
            self.get_logger().error(f"Color publishing error: {str(e)}")
    
    def publish_depth_frame(self, depth_image, predictions):
        """Process and publish depth frame with segmentation overlay"""
        try:
            # Get segmentation data
            panoptic_seg = predictions.get("panoptic_seg", None)
            if panoptic_seg is None:
                return
                
            # Create a colorized version of the depth image based on the segmentation
            # (simplified - in practice would use segment IDs to color the depth map)
            colored_depth = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Publish processed depth
            ros_image = self.bridge.cv2_to_imgmsg(colored_depth, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = self.depth_frame_id
            self.depth_publisher.publish(ros_image)
            self.get_logger().debug("Published depth segmentation result")
        except Exception as e:
            self.get_logger().error(f"Depth publishing error: {str(e)}")
    
    def draw_instance_boxes(self, image, predictions):
        """Draw bounding boxes, labels and distance information for object instances"""
        if "instances" not in predictions:
            return image
            
        instances = predictions["instances"].to("cpu")
        if len(instances) == 0:
            return image
        
        # Get boxes, classes and scores
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
        scores = instances.scores.numpy() if instances.has("scores") else None
        
        # Get depths if available
        instance_depths = None
        filtered_instances_idx = None
        if "depth_info" in predictions:
            instance_depths = predictions["depth_info"].get("instance_depths", None)
            filtered_instances_idx = predictions["depth_info"].get("filtered_instances_idx", None)
        
        # Get metadata for class names
        try:
            metadata = MetadataCatalog.get(self.demo.cfg.DATASETS.TEST[0] 
                         if len(self.demo.cfg.DATASETS.TEST) else "__unused")
            class_names = metadata.get("thing_classes", None)
        except:
            class_names = None
        
        # Make a copy of the image for drawing
        output_image = image.copy()
        
        if boxes is not None and len(boxes) > 0:
            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                # Use consistent colors for classes
                # Use class ID as seed for random generator to get consistent colors
                random.seed(int(cls))
                color = tuple([int(c) for c in np.random.randint(0, 255, size=3)])
                random.seed()  # Reset the seed
                
                # Convert box to int for drawing
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class name, score, and distance if available
                if class_names is not None and cls < len(class_names):
                    label = f"{class_names[cls]}: {score:.2f}"
                else:
                    label = f"Class {cls}: {score:.2f}"
                    
                # Add distance information if available
                if instance_depths is not None and filtered_instances_idx is not None:
                    try:
                        # Find this instance in the filtered instances
                        depth_idx = filtered_instances_idx.index(i) if i in filtered_instances_idx else -1
                        if depth_idx >= 0 and depth_idx < len(instance_depths):
                            # Add distance to label
                            distance = instance_depths[depth_idx]
                            label += f" | {distance:.2f}m"
                            
                            # Add a distance indicator line (longer = farther)
                            line_length = min(int(distance * 20), x2-x1-10)
                            line_start = (x1 + 5, y1 - 15)
                            line_end = (x1 + 5 + line_length, y1 - 15)
                            cv2.line(output_image, line_start, line_end, color, 2)
                    except (ValueError, IndexError) as e:
                        # If there's an error finding the depth, just continue without it
                        self.get_logger().debug(f"Could not find depth for instance {i}: {e}")
                
                # Draw label background
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output_image, 
                              (x1, y1 - text_size[1] - 10), 
                              (x1 + text_size[0], y1), 
                              color, 
                              -1)  # Filled rectangle
                
                # Draw label text (white on colored background)
                cv2.putText(output_image, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_image
    
    def publish_test_color_pointcloud(self):
        """Publish a simple colored test pointcloud to verify visualization"""
        try:
            # Create a red-green grid pattern
            if self.latest_pointcloud is None:
                self.get_logger().warn("No pointcloud available for test coloring")
                return
                
            # Make a copy of the latest pointcloud
            test_cloud = PointCloud2(
                header=self.latest_pointcloud.header,
                height=self.latest_pointcloud.height,
                width=self.latest_pointcloud.width,
                is_bigendian=self.latest_pointcloud.is_bigendian,
                point_step=self.latest_pointcloud.point_step,
                row_step=self.latest_pointcloud.row_step,
                is_dense=self.latest_pointcloud.is_dense,
                fields=self.latest_pointcloud.fields,
                data=bytearray(self.latest_pointcloud.data)
            )
            
            # Find RGB field
            rgb_offset = None
            for field in test_cloud.fields:
                if field.name == 'rgb' or field.name == 'rgba':
                    rgb_offset = field.offset
                    break
            
            if rgb_offset is None:
                self.get_logger().error("No RGB field in pointcloud for test coloring")
                return
                
            # Reshape data for easier access
            cloud_data = np.frombuffer(test_cloud.data, dtype=np.uint8).reshape(-1, test_cloud.point_step)
            
            # Apply a simple checker pattern (alternating red and green)
            total_points = test_cloud.width * test_cloud.height
            for i in range(total_points):
                # Alternating red and green
                if i % 2 == 0:
                    r, g, b = 255, 0, 0  # Red
                else:
                    r, g, b = 0, 255, 0  # Green
                    
                # Pack RGB in PCL format
                rgb_val = (r << 16) | (g << 8) | b
                rgb_float = struct.unpack('f', struct.pack('I', rgb_val))[0]
                rgb_bytes = struct.pack('f', rgb_float)
                
                # Update RGB value
                cloud_data[i, rgb_offset:rgb_offset+4] = np.frombuffer(rgb_bytes, dtype=np.uint8)
            
            # Update data
            test_cloud.data = bytes(cloud_data.tobytes())
            
            # Publish test cloud
            self.points_publisher.publish(test_cloud)
            self.get_logger().info("Published TEST colored pointcloud (red-green pattern)")
            
            return True
        except Exception as e:
            self.get_logger().error(f"Test pointcloud error: {str(e)}")
            traceback.print_exc()
            return False
    
    def publish_test_segmented_pointcloud(self):
        """Publish a persistent segmented pointcloud with colors matching the image segmentation"""
        try:
            # Check if we have pointcloud and segmentation data
            has_new_data = False
            
            with self.points_lock:
                if self.latest_pointcloud is not None:
                    current_cloud = self.latest_pointcloud
                    has_new_data = True
                elif self.last_segmented_cloud is not None:
                    # Re-publish the last segmented cloud if no new data
                    self.points_publisher.publish(self.last_segmented_cloud)
                    self.get_logger().debug("Re-published previous segmented pointcloud")
                    return True
                else:
                    # No data available yet
                    return False

            with self.result_lock:
                if self.latest_panoptic_result is not None:
                    current_seg_data = self.latest_panoptic_result
                    has_new_data = True
                elif self.last_segment_info is not None:
                    # We have segment info but no new pointcloud - use cached segments with new cloud
                    current_seg_data = {'predictions': {'panoptic_seg': self.last_segment_info}}
                else:
                    # No segmentation data yet
                    return False
                    
            if has_new_data:
                # Get segmentation data
                if 'panoptic_seg' in current_seg_data['predictions']:
                    panoptic_seg = current_seg_data['predictions']['panoptic_seg'][0].cpu().numpy()
                    segments_info = current_seg_data['predictions']['panoptic_seg'][1]
                    
                    # Cache the segmentation data for future use
                    with self.segmentation_lock:
                        self.last_segment_info = (panoptic_seg, segments_info)
                        
                    # Make a copy of the pointcloud
                    test_cloud = PointCloud2(
                        header=current_cloud.header,
                        height=current_cloud.height,
                        width=current_cloud.width,
                        is_bigendian=current_cloud.is_bigendian,
                        point_step=current_cloud.point_step,
                        row_step=current_cloud.row_step,
                        is_dense=current_cloud.is_dense,
                        fields=current_cloud.fields,
                        data=bytearray(current_cloud.data)
                    )
                    
                    # Find RGB field
                    rgb_offset = None
                    for field in test_cloud.fields:
                        if field.name == 'rgb' or field.name == 'rgba':
                            rgb_offset = field.offset
                            break
                    
                    if rgb_offset is None:
                        self.get_logger().error("No RGB field in pointcloud for segmentation coloring")
                        return False
                            
                    # Reshape data for easier access
                    cloud_data = np.frombuffer(test_cloud.data, dtype=np.uint8).reshape(-1, test_cloud.point_step)
                    
                    # Get Detectron2 visualization colors - extract from the visualization output if available
                    segment_colors = {}
                    
                    # Method 1: Extract colors from visualization result if available
                    if 'vis_output' in current_seg_data:
                        # Get the visualization image
                        vis_image = current_seg_data['vis_output'].get_image()
                        
                        # Create a mapping from segment_id to color by sampling the visualization
                        # This needs to happen for each segment
                        for segment in segments_info:
                            segment_id = segment['id']
                            # Find pixels that have this segment ID
                            segment_mask = panoptic_seg == segment_id
                            if np.any(segment_mask):
                                # Sample a pixel from this segment to get the color
                                y, x = np.where(segment_mask)
                                if len(y) > 0 and len(x) > 0:
                                    # Get the color from the visualization image
                                    idx = len(y) // 2  # Take a pixel from the middle of the segment
                                    r, g, b = vis_image[y[idx], x[idx], :]
                                    segment_colors[segment_id] = (r, g, b)
                                else:
                                    # Fallback: Use a deterministic color based on category ID
                                    category_id = segment.get('category_id', 0)
                                    random.seed(category_id * 100)
                                    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                                    segment_colors[segment_id] = color
                            else:
                                # Fallback for segments not found in the image
                                category_id = segment.get('category_id', 0)
                                random.seed(category_id * 100)
                                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                                segment_colors[segment_id] = color
                    else:
                        # Fallback: Method 2 - Use the same color generation logic as Detectron2
                        metadata = MetadataCatalog.get(self.demo.cfg.DATASETS.TEST[0] 
                                     if len(self.demo.cfg.DATASETS.TEST) else "__unused")
                        
                        for segment in segments_info:
                            segment_id = segment['id']
                            category_id = segment.get('category_id', 0)
                            is_thing = segment.get('isthing', False)
                            
                            if is_thing:
                                # For "thing" classes, use a deterministic color based on instance ID
                                random.seed(segment_id)
                                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                            else:
                                # For "stuff" classes, use a color based on category ID
                                # This mimics how Detectron2 colors stuff classes
                                random.seed(category_id)
                                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                            
                            segment_colors[segment_id] = color
                    
                    # Reset the random seed
                    random.seed(None)
                    
                    # Apply colors to pointcloud
                    total_points = test_cloud.width * test_cloud.height
                    segmentation_height = panoptic_seg.shape[0]
                    segmentation_width = panoptic_seg.shape[1]
                    
                    for i in range(total_points):
                        # Map point to segment using direct mapping
                        row = (i // test_cloud.width) % segmentation_height
                        col = (i % test_cloud.width) % segmentation_width
                        
                        # Get segment ID at this position
                        segment_id = panoptic_seg[row, col]
                        
                        # Assign color based on segment
                        if segment_id in segment_colors:
                            r, g, b = segment_colors[segment_id]
                        else:
                            # Use a default color that stands out
                            r, g, b = 200, 200, 200
                            
                        # Pack RGB in PCL format
                        rgb_val = (r << 16) | (g << 8) | b
                        rgb_float = struct.unpack('f', struct.pack('I', rgb_val))[0]
                        rgb_bytes = struct.pack('f', rgb_float)
                        
                        # Update RGB value
                        cloud_data[i, rgb_offset:rgb_offset+4] = np.frombuffer(rgb_bytes, dtype=np.uint8)
                    
                    # Update data
                    test_cloud.data = bytes(cloud_data.tobytes())
                    
                    # Cache the segmented cloud
                    with self.segmentation_lock:
                        self.last_segmented_cloud = test_cloud
                    
                    # Publish cloud
                    self.points_publisher.publish(test_cloud)
                    self.get_logger().info("Published segmented pointcloud with matching image colors")
                    
                    return True
                else:
                    self.get_logger().warn("No panoptic segmentation found in latest results")
                    return False
                    
        except Exception as e:
            self.get_logger().error(f"Segmented pointcloud error: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def publish_laser_scan(self, predictions):
        """Convert instance detections to laser scan and publish with full mask contours"""
        try:
            # Check if we have the required data
            if "instances" not in predictions or "panoptic_seg" not in predictions:
                return
                
            # Create LaserScan message
            scan_msg = LaserScan()
            scan_msg.header.stamp = self.get_clock().now().to_msg()
            scan_msg.header.frame_id = "base_scan"
            
            # Set scan parameters
            angle_min = -math.pi/2   # Field of view left limit
            angle_max = math.pi/2    # Field of view right limit
            angle_increment = 0.01   # Angular resolution
            range_min = 0.1
            range_max = 10.0
            
            # Number of rays
            num_rays = int((angle_max - angle_min) / angle_increment) + 1
            ranges = [float('inf')] * num_rays
            
            # Get camera parameters
            if self.camera_info is not None:
                K = self.camera_info.k
                fx = K[0]  # Focal length x
                fy = K[4]  # Focal length y
                cx = K[2]  # Principal point x
                cy = K[5]  # Principal point y
                width = self.camera_info.width
                height = self.camera_info.height
            else:
                # Default values if camera info not available
                fx = 525.0
                fy = 525.0
                cx = 319.5
                cy = 239.5
                width = 640
                height = 480
            
            with self.depth_lock:
                depth_img = self.latest_depth_image
                if depth_img is None:
                    return
            
            # Get panoptic segmentation data
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            panoptic_seg = panoptic_seg.cpu().numpy()
            
            # Process "thing" instances using segmentation masks
            instances = predictions["instances"].to("cpu")
            if len(instances) > 0 and "depth_info" in predictions:
                instance_depths = predictions["depth_info"].get("instance_depths", [])
                filtered_instances_idx = predictions["depth_info"].get("filtered_instances_idx", [])
                
                for i, idx in enumerate(filtered_instances_idx):
                    if idx >= len(instances) or i >= len(instance_depths):
                        continue
                    
                    # Get the class ID and check if we want to include this object type
                    if instances.has("pred_classes"):
                        class_id = int(instances.pred_classes[idx].item())
                        # Optional: filter by class if needed
                        # if class_id not in [0, 1, 2]:  # Example: only include certain classes
                        #     continue
                    
                    # Get instance ID from segmentation info to match with panoptic_seg
                    instance_id = None
                    for segment in segments_info:
                        if segment.get('isthing', False) and segment.get('instance_id', -1) == idx:
                            instance_id = segment['id']
                            break
                    
                    if instance_id is None:
                        continue
                    
                    # Create mask for this instance
                    instance_mask = (panoptic_seg == instance_id)
                    
                    if not np.any(instance_mask):
                        continue
                    
                    # Find contours of the mask
                    # Convert mask to uint8 for findContours
                    mask_uint8 = np.uint8(instance_mask) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        continue
                    
                    # Use the largest contour
                    contour = max(contours, key=cv2.contourArea)
                    
                    # Sample points along the contour
                    # We'll use more points for larger contours
                    num_points = max(20, len(contour) // 5)
                    step = max(1, len(contour) // num_points)
                    
                    for j in range(0, len(contour), step):
                        # Get point from contour
                        pt = contour[j][0]
                        col, row = pt
                        
                        # Skip if out of bounds
                        if not (0 <= row < depth_img.shape[0] and 0 <= col < depth_img.shape[1]):
                            continue
                        
                        # Get depth for this point
                        depth_value = depth_img[row, col]
                        
                        # Skip invalid depths
                        if depth_value <= 0.1 or depth_value > 10.0:
                            continue
                        
                        # Convert to camera coordinates
                        z = depth_value
                        x_cam = (col - cx) * z / fx
                        
                        # Calculate angle and distance
                        angle = math.atan2(x_cam, z)
                        distance = math.sqrt(x_cam*x_cam + z*z)
                        
                        # Map to LaserScan
                        if angle_min <= angle <= angle_max:
                            index = int((angle - angle_min) / angle_increment)
                            
                            # Update if closer
                            if 0 <= index < num_rays and distance < ranges[index]:
                                ranges[index] = max(range_min, min(distance, range_max))
            
            # Process "stuff" segments that are obstacles
            for segment in segments_info:
                if not segment.get('isthing', True):  # Focus on "stuff" classes
                    segment_id = segment['id']
                    category_id = segment.get('category_id', 0)
                    
                    # Get metadata for class names
                    metadata = MetadataCatalog.get(self.demo.cfg.DATASETS.TEST[0] 
                                     if len(self.demo.cfg.DATASETS.TEST) else "__unused")

                    # Check if this category represents an obstacle
                    is_obstacle = self.is_obstacle_class(category_id, metadata)
                    
                    if is_obstacle:
                        # Create a mask for this segment
                        segment_mask = (panoptic_seg == segment_id)
                        
                        if not np.any(segment_mask):
                            continue
                        
                        # Find contours of the mask
                        mask_uint8 = np.uint8(segment_mask) * 255
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if not contours:
                            continue
                        
                        # Use the largest contour
                        contour = max(contours, key=cv2.contourArea)
                        
                        # Sample points along the contour
                        num_points = max(20, len(contour) // 5)
                        step = max(1, len(contour) // num_points)
                        
                        for j in range(0, len(contour), step):
                            # Get point from contour
                            pt = contour[j][0]
                            col, row = pt
                            
                            # Skip if out of bounds
                            if not (0 <= row < depth_img.shape[0] and 0 <= col < depth_img.shape[1]):
                                continue
                            
                            # Get depth for this point
                            depth_value = depth_img[row, col]
                            
                            # Skip invalid depths
                            if depth_value <= 0.1 or depth_value > 10.0:
                                continue
                            
                            # Convert to camera coordinates
                            z = depth_value
                            x_cam = (col - cx) * z / fx
                            
                            # Calculate angle and distance
                            angle = math.atan2(x_cam, z)
                            distance = math.sqrt(x_cam*x_cam + z*z)
                            
                            # Map to LaserScan
                            if angle_min <= angle <= angle_max:
                                index = int((angle - angle_min) / angle_increment)
                                
                                # Update if closer
                                if 0 <= index < num_rays and distance < ranges[index]:
                                    ranges[index] = max(range_min, min(distance, range_max))
            
            # Fill small gaps in the scan for more continuous obstacles
            filled_ranges = self.fill_scan_gaps(ranges)
            
            # Set LaserScan message fields
            scan_msg.angle_min = angle_min
            scan_msg.angle_max = angle_max
            scan_msg.angle_increment = angle_increment
            scan_msg.time_increment = 0.0
            scan_msg.scan_time = 0.1
            scan_msg.range_min = range_min
            scan_msg.range_max = range_max
            scan_msg.ranges = filled_ranges
            
            # Publish
            self.scan_publisher.publish(scan_msg)
            self.get_logger().debug("Published enhanced laser scan using full segmentation masks")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing laser scan: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def fill_scan_gaps(self, ranges, max_gap_width=5):
        """Fill small gaps in the laser scan data for more continuous obstacle detection"""
        filled_ranges = list(ranges)
        
        # Find gaps (sequences of inf values)
        i = 0
        while i < len(filled_ranges):
            if math.isinf(filled_ranges[i]):
                # Found the start of a gap
                gap_start = i
                
                # Find the end of the gap
                while i < len(filled_ranges) and math.isinf(filled_ranges[i]):
                    i += 1
                
                gap_end = i - 1
                gap_width = gap_end - gap_start + 1
                
                # Fill small gaps by linear interpolation
                if gap_width <= max_gap_width and gap_start > 0 and gap_end < len(filled_ranges) - 1:
                    left_val = filled_ranges[gap_start - 1]
                    right_val = filled_ranges[gap_end + 1]
                    
                    # Only interpolate if both sides are valid ranges
                    if not math.isinf(left_val) and not math.isinf(right_val):
                        for j in range(gap_start, gap_end + 1):
                            # Linear interpolation
                            alpha = (j - gap_start + 1) / (gap_width + 1)
                            filled_ranges[j] = left_val * (1 - alpha) + right_val * alpha
            else:
                i += 1
                
        return filled_ranges
    
    def is_obstacle_class(self, category_id, metadata):
        """Determine if a category should be considered an obstacle based on class name"""
        # Classes to ignore (not obstacles)
        ignore_classes = [
            "sky", "road", "floor", "ground", "terrain", "grass", "earth", "field", 
            "path", "pavement", "dirt", "gravel", "carpet", "mat", "rug"
        ]
        
        # Get class name from metadata
        try:
            if hasattr(metadata, "stuff_classes"):
                # Get the class index within stuff classes
                # This requires mapping from the category_id to the stuff_dataset_id_to_contiguous_id
                if hasattr(metadata, "stuff_dataset_id_to_contiguous_id"):
                    if category_id in metadata.stuff_dataset_id_to_contiguous_id:
                        class_idx = metadata.stuff_dataset_id_to_contiguous_id[category_id]
                        if class_idx < len(metadata.stuff_classes):
                            class_name = metadata.stuff_classes[class_idx].lower()
                            return class_name not in ignore_classes
            
            # Fallback to category ID-based filtering if we can't get the class name
            # These are common COCO stuff category IDs to ignore
            ignore_category_ids = [40, 92, 96, 123, 147, 148, 149, 155, 170]
            return category_id not in ignore_category_ids
        except:
            # If anything goes wrong, default to assuming it's an obstacle
            self.get_logger().warn(f"Error determining if category {category_id} is an obstacle, treating as obstacle")
            return True
    
    def shutdown(self):
        """Clean shutdown of the node"""
        self.running = False
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        if hasattr(self, 'points_thread') and self.points_thread.is_alive():
            self.points_thread.join(timeout=1.0)
        
        # Clean up OpenCV windows properly
        if self.visualize:
            try:
                cv2.destroyAllWindows()
            except:
                pass
                
        self.get_logger().info("Node shutting down")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ROS-enabled Detectron2 Panoptic Segmentation")
    parser.add_argument(
        "--config-file",
        default="../configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for predictions to be shown",
    )
    parser.add_argument(
        "--color-topic",
        default="/a200_1093/sensors/camera_0/color/image",
        help="ROS topic for color images",
    )
    parser.add_argument(
        "--depth-topic",
        default="/a200_1093/sensors/camera_0/depth/image",
        help="ROS topic for depth images",
    )
    parser.add_argument(
        "--points-topic",
        default="/a200_1093/sensors/camera_0/points",
        help="ROS topic for point cloud data",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable visualization window",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    # Create node
    node = PanopticSegmentationNode(
        config_file=args.config_file,
        weights=args.weights,
        confidence_threshold=args.confidence_threshold,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        points_topic=args.points_topic,
        visualize=not args.no_visualization
    )
    
    # Set log level if debug is enabled
    if args.debug:
        node.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
    
    # Spin node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()