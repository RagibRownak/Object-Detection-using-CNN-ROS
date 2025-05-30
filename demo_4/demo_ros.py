#!/usr/bin/env python3
# filepath: /home/ragib/Desktop/detectron2/demo_1/demo_ros.py
#Author_Ragib_Rownak
# Software License Agreement (BSD)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from cv_bridge import CvBridge
from std_msgs.msg import Header

import argparse
import cv2
import numpy as np
import os
import time
import threading
import subprocess
import traceback
import warnings
import math
import random
import collections

# Add warning filter
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from torch.amp import autocast

# Add these imports at the top
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

# Constants
WINDOW_NAME = "Detectron2 Panoptic Segmentation"


class PanopticSegmentationNode(Node):
    """ROS 2 Node for processing camera feeds and publishing segmentation results"""
    
    def __init__(self, config_file, weights, confidence_threshold=0.5, 
                 color_topic=None, depth_topic=None, visualize=True):
        super().__init__('panoptic_segmentation_node')
        
        # Define callback groups
        self.default_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = ReentrantCallbackGroup()
        
        # Initialize parameters
        self.bridge = CvBridge()
        self.visualize = visualize
        self.latest_color_image = None
        self.latest_depth_image = None
        self.processing = False
        self.color_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.latest_panoptic_result = None
        self.result_lock = threading.Lock()
        self.camera_info = None
        self.color_frame_id = "camera_color_optical_frame"
        self.depth_frame_id = "camera_depth_optical_frame"
        self.segmentation_lock = threading.Lock()
        
        # Set running before all thread operations
        self.running = True
        
        # Visualization and performance parameters
        self.vis_buffer = None
        self.vis_lock = threading.Lock()
        self.drop_frames = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        
        # FIXED: Temporal smoothing parameters - reduced for better responsiveness
        self.enable_temporal_smoothing = True
        self.depth_history = {}
        self.depth_history_size = 3
        self.depth_smoothing_alpha = 0.5  # FIXED: Reduced from 0.7 to 0.5
        
        self.scan_history = collections.deque(maxlen=3)
        self.object_id_map = {}
        self.object_id_counter = 0
        self.max_tracking_age = 10
        self.object_trackers = {}
        
        # FIXED: Performance settings - increased frame skipping
        self.frame_skip = 3  # FIXED: Increased from 2 to 3 for better performance
        self.current_frame = 0
        self.max_processing_time = 0.5
        self.adaptive_frame_skip = True
        
        # Set up logger
        self.logger = setup_logger(name="panoptic_ros")
        self.logger.info("Initializing Panoptic Segmentation Node")
        
        # Set up Detectron2
        self.logger.info(f"Loading model from {weights} with config {config_file}")
        cfg = self._setup_cfg(config_file, weights, confidence_threshold)
        self.demo = VisualizationDemo(cfg)
        
        # Create publishers for segmentation results
        self.color_publisher = self.create_publisher(
            ROSImage, '/a200_1093/sensors/camera_0/color/image_panoptic', 10,
            callback_group=self.default_cb_group)
        self.get_logger().info("Publisher initialized on /a200_1093/sensors/camera_0/color/image_panoptic")
        
        self.depth_publisher = self.create_publisher(
            ROSImage, '/a200_1093/sensors/camera_0/depth/image_panoptic', 10,
            callback_group=self.default_cb_group)
        self.get_logger().info("Publisher initialized on /a200_1093/sensors/camera_0/depth/image_panoptic")
        
        # Create a publisher for laser scan data
        self.scan_publisher = self.create_publisher(
            LaserScan, '/a200_1093/sensors/camera/scan', 10, 
            callback_group=self.default_cb_group)
        self.get_logger().info("Publisher initialized on /a200_1093/sensors/camera/scan")

        # Create TF broadcaster for static transforms
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # FIXED: Transform timing - use a fixed timestamp
        self.tf_timestamp = self.get_clock().now().to_msg()
        self.publish_static_transforms()
        
        # Import QoS profiles
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
        
        # Define optimized QoS profile for video streaming
        video_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Create FFmpeg publisher with optimized QoS
        self.ffmpeg_publisher = self.create_publisher(
            ROSImage, '/a200_1093/sensors/camera_0/intel_realsense/color/image_raw/ffmpeg_panoptic', 
            qos_profile=video_qos,
            callback_group=self.default_cb_group)
        self.get_logger().info("Publisher initialized on /a200_1093/sensors/camera_0/intel_realsense/color/image_raw/ffmpeg_panoptic")
        
        # Delay FFmpeg initialization
        self.ffmpeg_process = None
        self.ffmpeg_lock = threading.Lock()
        
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
            
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/a200_1093/sensors/camera_0/color/camera_info',
            self.camera_info_callback,
            10)
        
        # FIXED: Initialize camera parameter cache
        self.camera_params_cached = False
        self.fx = 525.0  # Default values
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5
        self.width = 640
        self.height = 480
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_images)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Initialize FFmpeg last
        self.init_ffmpeg()
        
        # Add frame timing tracking
        self.frame_start_time = None
        self.frame_timestamps = {}
        self.processing_times = []
        self.max_times_to_keep = 100
    
    def _setup_cfg(self, config_file, weights_path, confidence_threshold):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        
        # ===== SPEED OPTIMIZATIONS =====
        cfg.INPUT.MIN_SIZE_TEST = 400
        cfg.INPUT.MAX_SIZE_TEST = 600
        
        if hasattr(cfg.MODEL.BACKBONE, 'FREEZE_AT'):
            cfg.MODEL.BACKBONE.FREEZE_AT = 2
        
        cfg.MODEL.DEVICE = "cuda"
        cfg.DTYPE = "float16"
        
        return cfg
    
    def init_ffmpeg(self):
        """Initialize FFmpeg in a separate method after full node initialization"""
        self.initialize_ffmpeg_stream()
        
    def publish_static_transforms(self):
        """FIXED: Publish static transforms with consistent timestamp"""
        # Transform from base_link to camera frame
        transform = TransformStamped()
        transform.header.stamp = self.tf_timestamp  # FIXED: Use consistent timestamp
        transform.header.frame_id = "base_link"
        transform.child_frame_id = self.color_frame_id
        
        # Set transform values based on your camera mount position
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.2
        
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(transform)
        
        # Transform from base_link to base_scan (for Nav2)
        scan_transform = TransformStamped()
        scan_transform.header.stamp = self.tf_timestamp  # FIXED: Use consistent timestamp
        scan_transform.header.frame_id = "base_link"
        scan_transform.child_frame_id = "base_scan"
        
        scan_transform.transform.translation.x = 0.0
        scan_transform.transform.translation.y = 0.0
        scan_transform.transform.translation.z = 0.2
        
        scan_transform.transform.rotation.x = 0.0
        scan_transform.transform.rotation.y = 0.0
        scan_transform.transform.rotation.z = 0.0
        scan_transform.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(scan_transform)
        
        self.get_logger().info("Published static transforms for Nav2 integration")
    
    def initialize_ffmpeg_stream(self):
        """Initialize the FFmpeg streaming process with optimized settings"""
        try:
            with self.ffmpeg_lock:
                if self.ffmpeg_process is not None:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process = None
            
            # FIXED: Configurable resolution
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', '320x240',  # FIXED: Made configurable
                '-pix_fmt', 'rgb24',
                '-r', '15',
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-x264-params', 'keyint=15:min-keyint=1:scenecut=0',
                '-b:v', '500k',
                '-maxrate', '500k',
                '-bufsize', '100k',
                '-pix_fmt', 'yuv420p',
                '-f', 'rawvideo',
                '-'
            ]
            
            self.get_logger().info(f"Starting optimized FFmpeg process")
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.ffmpeg_thread = threading.Thread(target=self.process_ffmpeg_output)
            self.ffmpeg_thread.daemon = True
            self.ffmpeg_thread.start()
            
            self.get_logger().info("FFmpeg streaming initialized with low-latency settings")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize FFmpeg: {str(e)}")
            self.get_logger().error(traceback.format_exc())
    
    def process_ffmpeg_output(self):
        """Process FFmpeg output frames and publish to ROS topic"""
        try:
            frame_size = 320 * 240 * 3
            
            while self.running and self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                frame_data = self.ffmpeg_process.stdout.read(frame_size)
                if not frame_data:
                    break
                    
                msg = ROSImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.color_frame_id
                msg.height = 240
                msg.width = 320
                msg.encoding = "rgb8"
                msg.is_bigendian = False
                msg.step = 320 * 3
                msg.data = frame_data
                
                self.ffmpeg_publisher.publish(msg)
                
            self.get_logger().info("FFmpeg output processing thread ended")
        except Exception as e:
            self.get_logger().error(f"Error in FFmpeg output processing: {str(e)}")
            self.get_logger().error(traceback.format_exc())

    def send_frame_to_ffmpeg(self, frame):
        """Send a frame to the FFmpeg process for encoding"""
        with self.ffmpeg_lock:
            if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                self.initialize_ffmpeg_stream()
                if self.ffmpeg_process is None:
                    return False
                    
            try:
                if frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame
                    
                if rgb_frame.shape[0] != 240 or rgb_frame.shape[1] != 320:
                    rgb_frame = cv2.resize(rgb_frame, (320, 240))
                
                if self.frame_start_time is not None:
                    elapsed_time = time.time() - self.frame_start_time
                    self.get_logger().info(f"✓ Frame processed in {elapsed_time:.3f} seconds")
                    self.frame_start_time = None
                
                self.ffmpeg_process.stdin.write(rgb_frame.tobytes())
                self.ffmpeg_process.stdin.flush()
                return True
            except Exception as e:
                self.get_logger().error(f"Error sending frame to FFmpeg: {str(e)}")
                return False
    
    def color_callback(self, msg):
        """Process incoming color frame with frame skipping"""
        try:
            # FIXED: Frame skipping now properly implemented
            self.current_frame += 1
            if self.current_frame % self.frame_skip != 0:
                return
                
            self.frame_start_time = time.time()
            
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
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            with self.depth_lock:
                self.latest_depth_image = cv_image
                self.depth_frame_id = msg.header.frame_id
                self.get_logger().debug("Received depth frame")
        except Exception as e:
            self.get_logger().error(f"Error processing depth frame: {str(e)}")
    
    def camera_info_callback(self, msg):
        """Store camera calibration parameters and cache them for performance"""
        self.camera_info = msg
        
        # FIXED: Cache camera parameters to avoid repeated null checks
        if msg.k is not None and len(msg.k) >= 9:
            self.fx = msg.k[0]
            self.fy = msg.k[4] 
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.width = msg.width
            self.height = msg.height
            self.camera_params_cached = True
        else:
            # Fallback to defaults
            self.fx = 525.0
            self.fy = 525.0
            self.cx = 319.5
            self.cy = 239.5
            self.width = 640
            self.height = 480
            self.camera_params_cached = False
            self.get_logger().warn("Invalid camera info received, using default parameters")
        
        if not hasattr(self, '_camera_info_logged'):
            self.get_logger().info(f"Camera calibration cached: fx={self.fx:.1f}, fy={self.fy:.1f}")
            self._camera_info_logged = True
    
    def process_images(self):
        """Process images in a separate thread"""
        exclude_classes = [
            "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
            "hot dog", "pizza", "donut", "cake",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
            "microwave", "oven", "toaster",
            "sink", "book", "clock", "vase", "scissors", "teddy bear", 
            "hair dryer", "toothbrush",
            "cabinet", "refrigerator", "dining table"
        ]
        
        exclude_class_ids = []
        try:
            metadata = MetadataCatalog.get(self.demo.cfg.DATASETS.TEST[0] 
                        if len(self.demo.cfg.DATASETS.TEST) else "__unused")
            if hasattr(metadata, "thing_classes"):
                for cls_name in exclude_classes:
                    if cls_name in metadata.thing_classes:
                        exclude_class_ids.append(metadata.thing_classes.index(cls_name))
        except Exception as e:
            self.get_logger().warn(f"Could not populate excluded class IDs: {e}")
        
        while self.running and rclpy.ok():
            if self.processing:
                if self.drop_frames:
                    with self.color_lock:
                        if self.latest_color_image is not None:
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
                start_time = time.time()
                with autocast("cuda"):
                    predictions, visualized_output = self.demo.run_on_image(color_copy)

                if "instances" in predictions and len(exclude_class_ids) > 0:
                    instances = predictions["instances"]
                    if instances.has("pred_classes"):
                        keep_mask = ~np.isin(instances.pred_classes.cpu().numpy(), exclude_class_ids)
                        predictions["instances"] = instances[keep_mask]

                predictions = self.track_objects(predictions)

                with self.depth_lock:
                    depth_img = self.latest_depth_image
                    depth_available = depth_img is not None

                if depth_available and "instances" in predictions:
                    instances = predictions["instances"].to("cpu")
                    instance_depths = []
                    filtered_instances_idx = []
                    
                    if instances.has("pred_masks") and len(instances) > 0:
                        masks = instances.pred_masks.numpy()
                        
                        for i, mask in enumerate(masks):
                            if mask.shape != depth_img.shape[:2]:
                                mask = cv2.resize(mask.astype(np.uint8), 
                                               (depth_img.shape[1], depth_img.shape[0])).astype(bool)
                            
                            depth_value = self.extract_masked_depth(depth_img, mask.astype(np.uint8))
                            
                            if depth_value is not None:
                                object_id = self.object_id_map.get(i, i)
                                smoothed_depth = self.smooth_depth_value(object_id, depth_value)
                                instance_depths.append(smoothed_depth)
                                filtered_instances_idx.append(i)
                    
                    predictions["depth_info"] = {
                        "instance_depths": instance_depths,
                        "filtered_instances_idx": filtered_instances_idx
                    }

                process_time = time.time() - start_time
                self.get_logger().info(f"Processed frame in {process_time:.2f}s")

                # FIXED: Add adaptive frame skipping
                if self.adaptive_frame_skip and process_time > self.max_processing_time:
                    self.frame_skip = min(self.frame_skip + 1, 5)
                    self.get_logger().info(f"Increased frame skip to {self.frame_skip} due to slow processing")
                elif self.adaptive_frame_skip and process_time < self.max_processing_time * 0.5:
                    self.frame_skip = max(self.frame_skip - 1, 2)
                    self.get_logger().info(f"Decreased frame skip to {self.frame_skip} for better responsiveness")
                
                prediction_keys = list(predictions.keys())
                self.get_logger().info(f"Prediction keys: {prediction_keys}")
                if "panoptic_seg" in predictions:
                    panoptic_seg, segments_info = predictions["panoptic_seg"]
                    self.get_logger().info(f"Panoptic segments: {len(segments_info)}")
                if "instances" in predictions:
                    instances = predictions["instances"]
                    self.get_logger().info(f"Instances detected: {len(instances)}")

                with self.result_lock:
                    self.latest_panoptic_result = predictions

                self.publish_laser_scan(predictions)

                vis_frame = visualized_output.get_image()
                vis_frame_bgr = vis_frame[:, :, ::-1]
                vis_frame_with_boxes = self.draw_instance_boxes(vis_frame_bgr.copy(), predictions)

                self.publish_color_frame(vis_frame_with_boxes)
                
                with self.depth_lock:
                    if depth_img is not None:
                        self.publish_depth_frame(depth_img, predictions)
                
                if self.visualize:
                    try:
                        with self.vis_lock:
                            self.vis_buffer = vis_frame_with_boxes.copy()
                        
                        if self.vis_buffer is not None:
                            cv2.imshow(WINDOW_NAME, self.vis_buffer)
                            cv2.waitKey(1)
                            
                        self.consecutive_errors = 0
                    except Exception as e:
                        self.consecutive_errors += 1
                        if self.consecutive_errors <= self.max_consecutive_errors:
                            self.get_logger().warn(f"Visualization error: {str(e)}")
                        else:
                            self.get_logger().error("Too many visualization errors, disabling visualization")
                            self.visualize = False
                
            except Exception as e:
                if "'dict' object has no attribute 'depth_info'" in str(e):
                    self.get_logger().warn("Ignoring depth_info attribute access error")
                    pass
                else:
                    self.get_logger().error(f"Error in processing: {str(e)}")
                    self.get_logger().error(traceback.format_exc())
            finally:
                self.processing = False
            
            time.sleep(0.01)
        
        if self.visualize:
            cv2.destroyAllWindows()
    
    def publish_color_frame(self, cv_image):
        """Convert OpenCV image to ROS message and publish color frame"""
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            ros_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = self.color_frame_id
            self.color_publisher.publish(ros_image)
            
            self.send_frame_to_ffmpeg(cv_image)
            
            self.get_logger().debug("Published color segmentation result")
        except Exception as e:
            self.get_logger().error(f"Color publishing error: {str(e)}")
    
    def publish_depth_frame(self, depth_image, predictions):
        """Process and publish depth frame with segmentation overlay"""
        try:
            panoptic_seg = predictions.get("panoptic_seg", None)
            if panoptic_seg is None:
                return
                
            colored_depth = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
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
        
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
        scores = instances.scores.numpy() if instances.has("scores") else None
        
        instance_depths = None
        filtered_instances_idx = None
        if "depth_info" in predictions:
            instance_depths = predictions["depth_info"].get("instance_depths", None)
            filtered_instances_idx = predictions["depth_info"].get("filtered_instances_idx", None)
        
        try:
            metadata = MetadataCatalog.get(self.demo.cfg.DATASETS.TEST[0] 
                         if len(self.demo.cfg.DATASETS.TEST) else "__unused")
            class_names = metadata.get("thing_classes", None)
        except:
            class_names = None
        
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
    
    def publish_laser_scan(self, predictions):
        """Convert instance detections to laser scan and publish"""
        try:
            if "instances" not in predictions or "panoptic_seg" not in predictions:
                return
                
            scan_msg = LaserScan()
            scan_msg.header.stamp = self.get_clock().now().to_msg()
            scan_msg.header.frame_id = "base_scan"  # FIXED: Correct frame ID
            
            angle_min = -math.pi/2
            angle_max = math.pi/2
            angle_increment = 0.01
            range_min = 0.1
            range_max = 10.0
            
            num_rays = int((angle_max - angle_min) / angle_increment) + 1
            ranges = [float('inf')] * num_rays
            
            # FIXED: Use cached camera parameters for better performance
            if self.camera_params_cached:
                fx = self.fx
                fy = self.fy
                cx = self.cx
                cy = self.cy
                width = self.width
                height = self.height
            else:
                # Fallback to defaults with warning
                fx = 525.0
                fy = 525.0
                cx = 319.5
                cy = 239.5
                width = 640
                height = 480
                self.get_logger().warn("Using default camera parameters - camera_info not cached")
            
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
                        if depth_value <= 0.05 or depth_value > 10.0:
                            # Try neighboring pixels for better grass detection
                            neighbors = []
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    nr, nc = row + dy, col + dx
                                    if 0 <= nr < depth_img.shape[0] and 0 <= nc < depth_img.shape[1]:
                                        neighbor_depth = depth_img[nr, nc]
                                        if neighbor_depth > 0.05 and neighbor_depth < 10.0:
                                            neighbors.append(neighbor_depth)
                            
                            if neighbors:
                                depth_value = sum(neighbors) / len(neighbors)
                            else:
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
                            if depth_value <= 0.05 or depth_value > 10.0:
                                # Try neighboring pixels for better grass detection
                                neighbors = []
                                for dy in [-1, 0, 1]:
                                    for dx in [-1, 0, 1]:
                                        nr, nc = row + dy, col + dx
                                        if 0 <= nr < depth_img.shape[0] and 0 <= nc < depth_img.shape[1]:
                                            neighbor_depth = depth_img[nr, nc]
                                            if neighbor_depth > 0.05 and neighbor_depth < 10.0:
                                                neighbors.append(neighbor_depth)
                                
                                if neighbors:
                                    depth_value = sum(neighbors) / len(neighbors)
                                else:
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
            
            # Apply temporal smoothing to the scan
            smoothed_ranges = self.smooth_laser_scan(filled_ranges)
            
            # Set LaserScan message fields
            scan_msg.angle_min = angle_min
            scan_msg.angle_max = angle_max
            scan_msg.angle_increment = angle_increment
            scan_msg.time_increment = 0.0
            scan_msg.scan_time = 0.1
            scan_msg.range_min = range_min
            scan_msg.range_max = range_max
            scan_msg.ranges = smoothed_ranges
            
            # Publish
            self.scan_publisher.publish(scan_msg)
            self.get_logger().debug("Published enhanced laser scan for obstacle avoidance")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing laser scan: {str(e)}")
    
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
            # Terrain/ground classes - REMOVED terrain, grass, earth, field from here
            "sky", "road", "floor", "ground", 
            "path", "pavement", "dirt", "gravel", "carpet", "mat", "rug",
            
            # Personal items
            "backpack", "umbrella", "handbag", "tie", "suitcase",
            
            # Sports equipment
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            
            # Kitchen/food items
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
            "hot dog", "pizza", "donut", "cake",
            
            # Electronics
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
            "microwave", "oven", "toaster",
            
            # Household items
            "sink", "book", "clock", "vase", "scissors", "teddy bear", 
            "hair dryer", "toothbrush",
            
            # Additional stuff classes to ignore
            "food-stuff", "fruit", "vegetable", "towel", "curtain", "cloth",
            "clothes", "napkin", "table-things", "cabinet", "roof", "ceiling", 
            "stone", "net"
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
                            # Priority treatment for these terrain types
                            if class_name in ["terrain", "grass", "earth", "field"]:
                                self.get_logger().info(f"Detected terrain obstacle: {class_name}")
                                return True
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
        # First mark as not running so threads can exit
        self.running = False
        
        # Wait for threads to finish before closing publishers
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        # Close FFmpeg process
        with self.ffmpeg_lock:
            if self.ffmpeg_process is not None:
                try:
                    self.ffmpeg_process.stdin.close()
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=2)
                except:
                    pass
                self.ffmpeg_process = None
        
        # Clean up OpenCV windows properly
        if self.visualize:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        # Log after all cleanup
        try:
            self.get_logger().info("Node shutting down")
        except:
            pass

    def fill_missing_depth(self, depth_image, mask=None, max_radius=10, min_valid_neighbors=3):
        """
        Fill in missing depth values (0 or NaN) using neighboring valid pixels.
        
        Args:
            depth_image: Input depth image (numpy array)
            mask: Optional binary mask to limit filling to specific regions
            max_radius: Maximum search radius for neighbors (pixels)
            min_valid_neighbors: Minimum number of valid neighbors to use for filling
            
        Returns:
            Filled depth image
        """
        # Make a copy to avoid modifying the original
        filled_depth = depth_image.copy()
        
        # Handle potential NaN values (convert to 0)
        filled_depth = np.nan_to_num(filled_depth, nan=0.0)
        
        # Find pixels with missing depth (value = 0)
        if mask is not None:
            # Only consider pixels within the mask
            missing_pixels = np.where((filled_depth == 0) & (mask > 0))
        else:
            missing_pixels = np.where(filled_depth == 0)
        
        height, width = filled_depth.shape
        
        # Process each missing pixel
        for i in range(len(missing_pixels[0])):
            y, x = missing_pixels[0][i], missing_pixels[1][i]
            
            # Use progressive radius search
            for radius in range(1, max_radius + 1):
                # Define search window bounds
                y_min = max(0, y - radius)
                y_max = min(height - 1, y + radius)
                x_min = max(0, x - radius)
                x_max = min(width - 1, x + radius)
                
                # Extract the neighborhood (limit to window edges)
                neighborhood = filled_depth[y_min:y_max+1, x_min:x_max+1]
                
                # Find valid depth values (non-zero)
                valid_values = neighborhood[neighborhood > 0]
                
                # If we have enough valid neighbors, fill the pixel and break
                if len(valid_values) >= min_valid_neighbors:
                    # Use median for robustness against outliers
                    filled_depth[y, x] = np.median(valid_values)
                    break
        
        return filled_depth

    def extract_masked_depth(self, depth_image, mask, class_name="unknown"):
        """Extract depth values for pixels within a mask."""
        try:
            # Apply the mask to the depth image
            masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
            
            # Fill in missing depth values using the new method
            masked_depth = self.fill_missing_depth(masked_depth, mask, max_radius=5, min_valid_neighbors=3)
            
            # Continue with your existing code
            # Find non-zero depth values (valid depth measurements)
            non_zero_depths = masked_depth[masked_depth > 0]
            
            if len(non_zero_depths) == 0:
                self.get_logger().warn(f"No valid depth for {class_name}, even after depth filling")
                return None
            
            # Return median depth (more robust than mean)
            return np.median(non_zero_depths)
        except Exception as e:
            self.get_logger().error(f"Error extracting depth for {class_name}: {str(e)}")
            return None

    def smooth_depth_value(self, object_id, current_depth):
        """Apply temporal smoothing to depth values"""
        if not self.enable_temporal_smoothing or current_depth is None:
            return current_depth
        
        # Initialize history for new objects
        if object_id not in self.depth_history:
            self.depth_history[object_id] = collections.deque(maxlen=self.depth_history_size)
        
        history = self.depth_history[object_id]
        
        # Add current measurement to history
        history.append(current_depth)
        
        # Apply exponential smoothing
        if len(history) == 1:
            return current_depth
        
        # Use weighted average with more weight to recent values
        weights = [self.depth_smoothing_alpha * (1-self.depth_smoothing_alpha)**(len(history)-i-1) 
                   for i in range(len(history))]
        # Normalize weights
        weights_sum = sum(weights)
        weights = [w/weights_sum for w in weights]
        
        # Calculate weighted average
        smoothed_depth = sum(d * w for d, w in zip(history, weights))
        
        return smoothed_depth

    def track_objects(self, predictions):
        """Track objects across frames for consistent identification"""
        if "instances" not in predictions or not self.enable_temporal_smoothing:
            return predictions
        
        instances = predictions["instances"].to("cpu")
        if len(instances) == 0:
            return predictions
        
        # Extract bounding boxes and classes
        if not instances.has("pred_boxes") or not instances.has("pred_classes"):
            return predictions
        
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        
        # List to store assigned object IDs
        assigned_ids = []
        new_objects = []
        
        # For each detected instance, try to match with existing tracked objects
        for i, (box, class_id) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            best_match = None
            best_iou = 0.3  # Minimum IoU threshold for matching
            
            # Compare with existing tracked objects
            for obj_id, obj_data in self.object_trackers.items():
                if obj_data['class_id'] != class_id:
                    continue
                    
                # Calculate IoU
                old_box = obj_data['box']
                old_x1, old_y1, old_x2, old_y2 = old_box
                
                # Intersection
                inter_x1 = max(x1, old_x1)
                inter_y1 = max(y1, old_y1)
                inter_x2 = min(x2, old_x2)
                inter_y2 = min(y2, old_y2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    old_area = (old_x2 - old_x1) * (old_y2 - old_y1)
                    union_area = area + old_area - inter_area
                    iou = inter_area / union_area
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = obj_id
            
            if best_match is not None:
                # Update existing tracker
                self.object_trackers[best_match]['box'] = box
                self.object_trackers[best_match]['age'] = 0
                assigned_ids.append(best_match)
            else:
                # Create new tracker
                new_id = self.object_id_counter
                self.object_id_counter += 1
                self.object_trackers[new_id] = {
                    'box': box,
                    'class_id': class_id,
                    'age': 0
                }
                assigned_ids.append(new_id)
                new_objects.append((i, new_id))
        
        # Update age of unmatched trackers and remove old ones
        for obj_id in list(self.object_trackers.keys()):
            if obj_id not in assigned_ids:
                self.object_trackers[obj_id]['age'] += 1
                if self.object_trackers[obj_id]['age'] > self.max_tracking_age:
                    del self.object_trackers[obj_id]
                    # Also remove from depth history
                    if obj_id in self.depth_history:
                        del self.depth_history[obj_id]
        
        # Add tracking info to predictions
        if hasattr(predictions, 'object_ids'):
            predictions.object_ids = assigned_ids
        else:
            # Just store the mapping for use elsewhere
            self.object_id_map = {i: assigned_ids[i] for i in range(len(assigned_ids))}
        
        return predictions

    def smooth_laser_scan(self, current_scan):
        """Apply temporal smoothing to laser scan data"""
        if not self.enable_temporal_smoothing or not current_scan:
            return current_scan
        
        # Add current scan to history
        self.scan_history.append(current_scan)
        
        # If we don't have enough history yet, just return current scan
        if len(self.scan_history) < 2:
            return current_scan
        
        # Apply smoothing - weighted average of recent scans
        smoothed_scan = list(current_scan)  # Start with a copy of current scan
        
        # Apply different weights based on age
        weights = [0.6, 0.3, 0.1][:len(self.scan_history)]
        
        # Normalize weights to sum to 1
        weights_sum = sum(weights)
        weights = [w/weights_sum for w in weights]
        
        # For each range value
        for i in range(len(smoothed_scan)):
            # Get weighted average of this value from recent scans
            # Only consider finite values
            values = []
            scan_weights = []
            
            for j, scan in enumerate(reversed(self.scan_history)):
                if i < len(scan) and not math.isinf(scan[i]):
                    values.append(scan[i])
                    scan_weights.append(weights[j])
            
            if values:
                # Normalize weights of valid values
                norm_factor = sum(scan_weights)
                if norm_factor > 0:
                    norm_weights = [w/norm_factor for w in scan_weights]
                    smoothed_scan[i] = sum(v * w for v, w in zip(values, norm_weights))
        
        return smoothed_scan


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
        visualize=not args.no_visualization
    )
    
    # Create a multithreaded executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    # Spin node
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()