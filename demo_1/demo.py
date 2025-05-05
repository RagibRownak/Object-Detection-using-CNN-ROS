# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import pyrealsense2 as rs  # Add RealSense import
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

# Add warning filter
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from torch.amp import autocast, GradScaler

# constants
WINDOW_NAME = "COCO detections"


class PanopticPublisher(Node):
    """ROS 2 Node for publishing panoptic segmentation results"""
    def __init__(self):
        super().__init__('panoptic_publisher')
        self.publisher = self.create_publisher(ROSImage, '/panoptic_segmentation', 10)
        self.bridge = CvBridge()
        self.frame_counter = 0
        self.get_logger().info("Panoptic segmentation publisher initialized")

    def publish_frame(self, cv_image):
        """Convert OpenCV image to ROS message and publish"""
        try:
            # Fix: Proper color space conversion for ROS
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            ros_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_color_optical_frame"
            self.publisher.publish(ros_image)
            
            self.frame_counter += 1
            if self.frame_counter % 5 == 0:
                rclpy.spin_once(self, timeout_sec=0.001)
                
        except Exception as e:
            self.get_logger().error(f"Publishing error: {str(e)}")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weights  # Set model weights
    print(f"Using model weights: {cfg.MODEL.WEIGHTS}")  # Debugging print statement
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 ROS-enabled panoptic segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml",  # Update to ResNet-50 config
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the model weights file",
    )
    parser.add_argument("--publish", action="store_true", help="Enable ROS2 publishing")
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=30,
            frameSize=(10, 10),
            isColor=True,
        )
        writer.release()
        return os.path.isfile(filename)


def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info(f"Launching with arguments: {args}")

    # Initialize ROS if publishing enabled
    ros_node = None
    if args.publish:
        rclpy.init()
        ros_node = PanopticPublisher()

    try:
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)
        scaler = GradScaler()

        if args.input:
            if len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"

            for path in tqdm.tqdm(args.input, disable=not args.output):
                img = read_image(path, format="BGR")
                start_time = time.time()
                with autocast("cuda"):
                    predictions, visualized_output = demo.run_on_image(img)
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        out_filename = args.output
                    visualized_output.save(out_filename)
                else:
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break
        elif args.webcam:
            # Initialize RealSense pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            
            try:
                while True:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    frame = np.asanyarray(color_frame.get_data())
                    predictions, vis_output = demo.run_on_image(frame)
                    
                    # Fix: Proper color conversion for visualization and publishing
                    vis_frame = cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)

                    if args.publish and ros_node:
                        ros_node.publish_frame(vis_frame)
                    
                    cv2.imshow(WINDOW_NAME, vis_frame)  # No need for additional conversion
                    if cv2.waitKey(1) == 27:
                        break

            finally:
                pipeline.stop()
                cv2.destroyAllWindows()
        elif args.video_input:
            video = cv2.VideoCapture(args.video_input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(args.video_input)
            codec, file_ext = ("mp4v", ".mp4") if test_opencv_video_format("mp4v", ".mp4") else ("DIVX", ".avi")
            if codec == "DIVX":
                warnings.warn("mp4v codec not available, switching to DIVX")
            if args.output:
                if os.path.isdir(args.output):
                    output_fname = os.path.join(args.output, basename)
                    output_fname = os.path.splitext(output_fname)[0] + file_ext
                else:
                    output_fname = args.output
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(args.video_input)
            batch_size = 15  # Increase batch size to 15
            frames = []
            for frame_idx in tqdm.tqdm(range(num_frames)):
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == batch_size or frame_idx == num_frames - 1:
                    start_time = time.time()
                    with autocast("cuda"):
                        predictions = demo.run_on_batch(frames)
                    logger.info(f"Processed batch of {len(frames)} frames in {time.time() - start_time:.2f}s")
                    for vis_frame in predictions:
                        if args.output:
                            output_file.write(vis_frame)
                        else:
                            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                            cv2.imshow(basename, vis_frame)
                            if cv2.waitKey(1) == 27:
                                break
                    frames = []
            video.release()
            if args.output:
                output_file.release()
            else:
                cv2.destroyAllWindows()

    finally:
        if args.publish and ros_node:
            ros_node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
