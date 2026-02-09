import depthai as dai
import cv2
import numpy as np
import yaml
import time
import logging
from pathlib import Path
from threading import Thread, Lock
from video_player import VideoPlayer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/detections.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SKUDetectionSystem:
    """
    Main system for SKU detection and content display
    """
    
    def __init__(self, config_path='../config/config.yaml'):
        logger.info("Initializing SKU Detection System...")
        
        self.config = self.load_config(config_path)
        
        self.player = VideoPlayer(
            resolution=tuple(self.config['display']['resolution']),
            fps=self.config['display']['fps'],
            fullscreen=self.config['display']['fullscreen'],
            fade_duration=self.config['transitions']['fade_duration']
        )

        # Load configuration parameters
        self.content_map = self.config['content']['skus']
        self.default_video = self.config['content']['default']
        self.confidence_threshold = self.config['detection']['confidence_threshold']
        self.model_path = self.config['detection']['model_path']
        self.labels_path = self.config['detection']['labels_path']
        self.labels = self.load_labels()
        
        # State tracking
        self.current_sku = None
        self.detection_lock = Lock()
        self.running = False
        
        # Preload videos if configured
        if self.config['video']['preload']:
            self.preload_videos()
        
        logger.info("System initialized successfully!")
    
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def load_labels(self):
        labels = []
        labels_path = Path(self.labels_path)
        
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(labels)} labels: {', '.join(labels)}")
        else:
            logger.warning(f"Labels file not found: {labels_path}")
        
        return labels
    
    def preload_videos(self):
        logger.info("Preloading videos...")
        
        # Preload default
        self.player.preload_video('default', self.default_video)
        
        # Preload SKU videos
        for sku, video_path in self.content_map.items():
            self.player.preload_video(sku, video_path)
        
        logger.info("All videos preloaded!")
    
    def create_detection_pipeline(self):
        logger.info("Creating detection pipeline...")
        
        pipeline = dai.Pipeline()
        
        # ================================================================
        # COLOR CAMERA
        # ================================================================
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 640)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(self.config['detection']['camera_fps'])
        
        # ================================================================
        # YOLO DETECTION NETWORK
        # ================================================================
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        detection_nn.setBlobPath(str(self.model_path))
        detection_nn.setConfidenceThreshold(self.confidence_threshold)
        detection_nn.setIouThreshold(0.5)
        
        # YOLOv8 settings
        detection_nn.setNumClasses(len(self.labels))
        detection_nn.setCoordinateSize(4)
        detection_nn.setAnchors([])
        detection_nn.setAnchorMasks({})
        detection_nn.setNumInferenceThreads(2)
        detection_nn.input.setBlocking(False)
        
        # Link camera to detection
        cam_rgb.preview.link(detection_nn.input)
        
        # ================================================================
        # OUTPUTS
        # ================================================================
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("detections")
        detection_nn.out.link(xout_nn.input)
        
        logger.info("Pipeline created successfully!")
        return pipeline
    
    def get_sku_from_detection(self, detections):
        if not detections:
            return None
        
        # Filter by confidence threshold
        valid_detections = [
            det for det in detections 
            if det.confidence >= self.confidence_threshold
        ]
        
        if not valid_detections:
            return None
        
        # Get highest confidence detection
        best_detection = max(valid_detections, key=lambda x: x.confidence)
        
        # Get label
        if best_detection.label < len(self.labels):
            sku = self.labels[best_detection.label]
            confidence = best_detection.confidence
            
            logger.debug(f"Detected: {sku} ({confidence:.2%})")
            return sku
        
        return None
    
    def handle_sku_change(self, new_sku):
        """Switch video content"""
        with self.detection_lock:
            if new_sku != self.current_sku:
                if new_sku is None:
                    # Return to default
                    logger.info("No product detected - showing default content")
                    self.player.switch_video('default', self.default_video)
                elif new_sku in self.content_map:
                    # Show SKU-specific content
                    video_path = self.content_map[new_sku]
                    logger.info(f"Detected {new_sku} - showing content: {video_path}")
                    self.player.switch_video(new_sku, video_path)
                else:
                    logger.warning(f"SKU {new_sku} not in content map - showing default")
                    self.player.switch_video('default', self.default_video)
                
                self.current_sku = new_sku
    
    def detection_loop(self, device):
        """Main detection loop - runs in separate thread"""
        logger.info("Starting detection loop...")
        
        # Get detection queue
        q_det = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        while self.running:
            try:
                # Get detections
                in_det = q_det.get()
                detections = in_det.detections
                
                # Process detections
                detected_sku = self.get_sku_from_detection(detections)
                
                # Handle SKU change
                self.handle_sku_change(detected_sku)
                
                # Small delay to prevent CPU spinning
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(0.1)
        
        logger.info("Detection loop stopped")
    
    def run(self):
        logger.info("="*70)
        logger.info("Starting SKU Detection and Display System")
        logger.info("="*70)
        
        # Verify model exists
        if not Path(self.model_path).exists():
            logger.error(f"Model not found: {self.model_path}")
            logger.error("Please ensure model is trained and converted to blob format")
            return
        
        # Verify videos exist
        if not Path(self.default_video).exists():
            logger.error(f"Default video not found: {self.default_video}")
            return
        
        # Create pipeline
        pipeline = self.create_detection_pipeline()
        
        # Start video player with default content
        logger.info("Starting video player...")
        self.player.start(self.default_video)
        
        # Connect to OAK-D Pro
        logger.info("Connecting to OAK-D Pro...")
        
        try:
            with dai.Device(pipeline) as device:
                logger.info(f"Connected to: {device.getDeviceName()}")
                logger.info(f"USB Speed: {device.getUsbSpeed().name}")
                
                # Start detection thread
                self.running = True
                detection_thread = Thread(
                    target=self.detection_loop,
                    args=(device,),
                    daemon=True
                )
                detection_thread.start()
                
                logger.info("="*70)
                logger.info("System running! Press 'q' to quit, 'r' to reset to default")
                logger.info("="*70)
                
                # Main loop - handle video playback and user input
                while True:
                    if not self.player.update():
                        break
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        logger.info("Quit requested by user")
                        break
                    elif key == ord('r'):
                        logger.info("Reset to default requested")
                        self.handle_sku_change(None)
                    elif key == ord('d'):
                        # Toggle debug info
                        self.player.show_debug = not self.player.show_debug
                        logger.info(f"Debug info: {self.player.show_debug}")
                
        except Exception as e:
            logger.error(f"Error running system: {e}")
            raise
        
        finally:
            # Cleanup
            logger.info("Shutting down...")
            self.running = False
            self.player.stop()
            cv2.destroyAllWindows()
            logger.info("Shutdown complete")


def main():
    Path('logs').mkdir(exist_ok=True)
    
    try:
        system = SKUDetectionSystem('../config/config.yaml')
        system.run()
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
