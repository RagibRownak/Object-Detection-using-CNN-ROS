#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import threading
import queue
import numpy as np
import os
import time
import warnings
import cv2
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import torch

warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo

# Constants
WINDOW_NAME = "Panoptic Segmentation"

class FrameProcessor(threading.Thread):
    """Thread that processes frames in the background"""
    def __init__(self, model, input_queue, output_queue, logger):
        threading.Thread.__init__(self)
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = logger
        self.daemon = True
        self.running = True
        
    def run(self):
        while self.running:
            try:
                # Get frame and frame ID from queue
                frame_id, frame = self.input_queue.get(timeout=1.0)
                if frame is None:
                    self.input_queue.task_done()
                    continue
                    
                # Process the frame
                start_time = time.time()
                try:
                    # Run inference - FIXED: Using proper format
                    predictions, vis_output = self.model.run_on_image(frame)
                    
                    # ADDED: Check if panoptic segmentation is present
                    if "panoptic_seg" in predictions:
                        panoptic_seg, segments_info = predictions["panoptic_seg"]
                        self.logger.info(f"Frame {frame_id}: Found {len(segments_info)} segments in panoptic seg")
                    else:
                        self.logger.warning(f"Frame {frame_id}: No panoptic segmentation in predictions")
                        for k in predictions.keys():
                            self.logger.info(f"  Available key: {k}")
                    
                    # Get visualization
                    vis_frame = vis_output.get_image()[:, :, ::-1]
                    inference_time = time.time() - start_time
                    fps = 1.0 / inference_time
                    self.logger.info(f"Processed frame {frame_id} in {inference_time:.3f}s ({fps:.1f} FPS)")
                    
                    # Put results in output queue
                    self.output_queue.put((frame_id, vis_frame, predictions, fps))
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_id}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())  # ADDED: More detailed error info
                    self.output_queue.put((frame_id, None, None, 0))
                finally:
                    self.input_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error in frame processor: {e}")
                import traceback
                self.logger.error(traceback.format_exc())  # ADDED: More detailed error info
        
    def stop(self):
        self.running = False

class PanopticPublisher(Node):
    def __init__(self):
        super().__init__('panoptic_publisher')
        self.publisher = self.create_publisher(ROSImage, '/panoptic_segmentation', 10)
        self.bridge = CvBridge()
        self.frame_counter = 0
        self.get_logger().info("Panoptic segmentation publisher initialized")

    def publish_frame(self, cv_image):
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            ros_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_color_optical_frame"
            self.publisher.publish(ros_image)
            
            self.frame_counter += 1
            if self.frame_counter % 3 == 0:
                rclpy.spin_once(self, timeout_sec=0.001)
        except Exception as e:
            self.get_logger().error(f"Publishing error: {str(e)}")

def setup_cfg(args):
    cfg = get_cfg()
    from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Performance optimizations
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Set weights
    cfg.MODEL.WEIGHTS = args.weights
    print(f"Using model weights: {cfg.MODEL.WEIGHTS}")
    
    # Set confidence threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    
    # ADDED: Fine-tuning the panoptic deeplab configuration
    if 'deeplab' in args.config_file.lower():
        # Ensure the panoptic head is properly configured
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19  # Cityscapes has 19 classes
        
        # Ensure correct input and output parameters
        if hasattr(cfg.MODEL, 'PANOPTIC_DEEPLAB'):
            # Make sure we're not using too small of a crop for training
            if hasattr(cfg.INPUT, 'CROP'):
                cfg.INPUT.CROP.SIZE = [512, 1024]
    
    # Apply any additional opts
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Panoptic Segmentation Demo")
    parser.add_argument("--config-file", default="configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml", help="path to config file")
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam")
    parser.add_argument("--video-input", help="Path to video file")
    parser.add_argument("--output", help="A file or directory to save output visualizations")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score for predictions")
    parser.add_argument("--opts", help="Additional config options", default=[], nargs=argparse.REMAINDER)
    parser.add_argument("--weights", required=True, help="Path to model weights file")
    parser.add_argument("--publish", action="store_true", help="Enable ROS2 publishing")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process 1 out of N+1 frames (0 means process all)")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of processing threads")
    parser.add_argument("--max-queue-size", type=int, default=5, help="Maximum size of frame queues")
    parser.add_argument("--display-size", type=str, default="1024x512", help="Display resolution (WxH)")
    parser.add_argument("--direct-display", action="store_true", help="Display segmentation directly without blending")
    return parser

def main():
    # Use spawn method for clean process creation
    mp.set_start_method("spawn", force=True)
    
    # Parse arguments
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info(f"Launching with arguments: {args}")
    
    # Parse display size
    try:
        display_width, display_height = map(int, args.display_size.split('x'))
    except:
        display_width, display_height = 1024, 512
        logger.warning(f"Invalid display size: {args.display_size}, using default: 1024x512")
    
    # Initialize ROS if publishing enabled
    ros_node = None
    if args.publish:
        try:
            rclpy.init()
            ros_node = PanopticPublisher()
        except Exception as e:
            logger.error(f"Failed to initialize ROS: {e}")
            args.publish = False

    try:
        # Set up model configuration
        cfg = setup_cfg(args)
        
        # Initialize model
        logger.info("Initializing model...")
        demo = VisualizationDemo(cfg)
        
        # ADDED: Test the model with a blank image to verify it works
        test_img = np.zeros((512, 1024, 3), dtype=np.uint8)
        logger.info("Testing model with blank image...")
        try:
            test_predictions, test_vis = demo.run_on_image(test_img)
            logger.info(f"Model test successful. Prediction keys: {list(test_predictions.keys())}")
            if "panoptic_seg" in test_predictions:
                logger.info("Panoptic segmentation is working correctly!")
            else:
                logger.warning("Model does not produce panoptic segmentation on test image!")
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Set up processing queues and threads
        input_queue = queue.Queue(maxsize=args.max_queue_size)
        output_queue = queue.Queue(maxsize=args.max_queue_size)
        
        # Create and start frame processor threads
        processors = []
        for i in range(max(1, min(args.num_threads, mp.cpu_count()))):
            processor = FrameProcessor(demo, input_queue, output_queue, logger)
            processor.start()
            processors.append(processor)
            logger.info(f"Started frame processor thread {i+1}")
        
        if args.webcam:
            # Try to initialize RealSense camera
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                # FIXED: Using exact 640x480 resolution that RealSense supports
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline.start(config)
                use_realsense = True
                logger.info("RealSense camera initialized")
            except Exception as e:
                logger.error(f"RealSense initialization failed: {e}")
                logger.info("Falling back to regular webcam")
                cap = cv2.VideoCapture(0)
                use_realsense = False

            # Frame tracking variables
            frame_count = 0
            processed_count = 0
            skip_count = args.skip_frames
            last_processed_frame = None
            last_fps = 0
            processing_times = []
            
            # Performance metrics
            display_fps = 0
            smoothing_factor = 0.1  # For EMA calculation
            last_time = time.time()
            
            try:
                while True:
                    loop_start = time.time()
                    
                    # Get frame from the appropriate camera
                    if use_realsense:
                        try:
                            frames = pipeline.wait_for_frames(1000)
                            color_frame = frames.get_color_frame()
                            if not color_frame:
                                continue
                            frame = np.asanyarray(color_frame.get_data())
                        except Exception as e:
                            logger.error(f"RealSense error: {e}")
                            continue
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                    
                    # Track frames 
                    frame_count += 1
                    process_this_frame = (skip_count == 0) or (frame_count % (skip_count + 1) == 0)
                    
                    # Update display FPS
                    current_time = time.time()
                    elapsed = current_time - last_time
                    if elapsed > 0:
                        current_display_fps = 1.0 / elapsed
                        display_fps = smoothing_factor * current_display_fps + (1 - smoothing_factor) * display_fps
                    last_time = current_time
                    
                    # Process frame if needed
                    if process_this_frame and input_queue.qsize() < args.max_queue_size:
                        # CRITICAL: Resize to EXACTLY 1024x512 as required by the model
                        processed_frame = cv2.resize(frame, (1024, 512))
                        
                        # ADDED: Make sure frame is in the correct format (BGR, uint8)
                        if processed_frame.dtype != np.uint8:
                            processed_frame = processed_frame.astype(np.uint8)
                        
                        # Put frame in processing queue
                        try:
                            input_queue.put((processed_count, processed_frame), block=False)
                            processed_count += 1
                        except queue.Full:
                            logger.warning("Input queue full, skipping frame")
                    
                    # Check for processed results
                    try:
                        while not output_queue.empty():
                            frame_id, vis_frame, predictions, fps = output_queue.get(block=False)
                            if vis_frame is not None:
                                # FIXED: Ensure visualization frame is correct
                                # Don't resize immediately - keep original resolution
                                last_processed_frame = vis_frame.copy()
                                last_fps = fps
                                
                                # Update processing time for metric
                                processing_times.append(1.0 / fps if fps > 0 else 0)
                                if len(processing_times) > 10:
                                    processing_times.pop(0)
                                
                                # Publish to ROS if enabled
                                if args.publish and ros_node:
                                    # FIXED: Publish the original visualization
                                    ros_node.publish_frame(vis_frame)
                            output_queue.task_done()
                    except queue.Empty:
                        pass
                    
                    # Prepare frame for display
                    if args.direct_display and last_processed_frame is not None:
                        # ADDED: Option to display segmentation directly
                        display_frame = cv2.resize(last_processed_frame, (display_width, display_height))
                    else:
                        # Normal display with camera view
                        display_frame = frame.copy()
                        
                        # If we have a processed frame, overlay it
                        if last_processed_frame is not None:
                            # FIXED: Improved visualization with stronger alpha
                            # Use alpha=1.0 for full visibility of segmentation
                            alpha = 1.0
                            processed_overlay = cv2.resize(last_processed_frame, (display_frame.shape[1], display_frame.shape[0]))
                            
                            # ADDED: Option to use more visible blending
                            blend_frame = np.zeros_like(display_frame)
                            cv2.addWeighted(processed_overlay, alpha, blend_frame, 1.0 - alpha, 0, blend_frame)
                            
                            # ADDED: Create a more visible mask for important segments
                            if alpha >= 0.7:
                                display_frame = processed_overlay.copy()
                            else:
                                cv2.addWeighted(processed_overlay, alpha, display_frame, 1.0 - alpha, 0, display_frame)
                    
                    # Add performance metrics
                    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
                    avg_processing_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    
                    # Create darker overlay for better text visibility
                    metrics_overlay = np.zeros((120, display_frame.shape[1], 3), dtype=np.uint8)
                    cv2.putText(metrics_overlay, f"Display FPS: {display_fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(metrics_overlay, f"Processing FPS: {avg_processing_fps:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(metrics_overlay, f"Process 1/{skip_count+1 if skip_count > 0 else 1} frames", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add overlay to top of image with darker blend
                    display_frame[:120, :] = cv2.addWeighted(display_frame[:120, :], 0.2, metrics_overlay, 0.8, 0)
                    
                    # Show the frame
                    cv2.imshow(WINDOW_NAME, display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC to exit
                        break
                    elif key == ord('+'): 
                        # Process more frames
                        skip_count = max(0, skip_count - 1)
                        logger.info(f"Processing 1/{skip_count+1 if skip_count > 0 else 1} frames")
                    elif key == ord('-'):
                        # Process fewer frames
                        skip_count = min(10, skip_count + 1)
                        logger.info(f"Processing 1/{skip_count+1} frames")
                    elif key == ord('d'):
                        # Toggle direct display mode
                        args.direct_display = not args.direct_display
                        logger.info(f"Direct display mode: {args.direct_display}")
                        
                    # Sleep to limit CPU usage if we're running very fast
                    loop_time = time.time() - loop_start
                    if loop_time < 0.01:  # Target 100 FPS for display loop
                        time.sleep(0.01 - loop_time)
            finally:
                # Stop processor threads
                for processor in processors:
                    processor.stop()
                
                # Clean up camera resources
                if use_realsense:
                    pipeline.stop()
                else:
                    cap.release()
                cv2.destroyAllWindows()
                
                # Wait for threads to finish
                for processor in processors:
                    processor.join(timeout=1.0)
        
        elif args.video_input:
            video = cv2.VideoCapture(args.video_input)
            frame_count = 0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                # CRITICAL: Resize to EXACTLY 1024x512 as required by the model
                processed_frame = cv2.resize(frame, (1024, 512))
                
                # Process frame
                try:
                    predictions, vis_output = demo.run_on_image(processed_frame)
                    
                    # Get visualization and resize back to original size
                    vis_frame = vis_output.get_image()[:, :, ::-1]
                    vis_frame = cv2.resize(vis_frame, (frame.shape[1], frame.shape[0]))
                    
                    if args.output:
                        # Save output if requested
                        output_path = os.path.join(args.output, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(output_path, vis_frame)
                    else:
                        cv2.imshow("video", vis_frame)
                        if cv2.waitKey(1) == 27:  # ESC to exit
                            break
                except Exception as e:
                    logger.error(f"Error processing video frame {frame_count}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                        
                frame_count += 1
            
            video.release()
            if not args.output:
                cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Clean up ROS resources
        if args.publish and ros_node:
            try:
                ros_node.destroy_node()
                rclpy.shutdown()
            except Exception as e:
                logger.error(f"ROS shutdown error: {e}")

if __name__ == "__main__":
    main()