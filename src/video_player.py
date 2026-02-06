"""
Video Player with Smooth Fade Transitions
Optimized for Raspberry Pi 5 hardware acceleration
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


class VideoPlayer:
    """
    Hardware-accelerated video player with smooth fade transitions
    """
    
    def __init__(self, resolution=(1920, 1080), fps=30, fullscreen=True, fade_duration=0.5):
        """
        Initialize video player
        
        Args:
            resolution: Display resolution (width, height)
            fps: Target FPS for display
            fullscreen: Whether to display fullscreen
            fade_duration: Duration of fade transition in seconds
        """
        self.resolution = resolution
        self.fps = fps
        self.fullscreen = fullscreen
        self.fade_duration = fade_duration
        self.fade_frames = int(fps * fade_duration)
        
        # Video state
        self.current_video = None
        self.current_cap = None
        self.next_video = None
        self.next_cap = None
        
        # Transition state
        self.transitioning = False
        self.transition_frame = 0
        
        # Frame cache for smooth playback
        self.frame_buffer = deque(maxlen=3)
        
        # Preloaded videos cache
        self.video_cache = {}
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.actual_fps = 0
        
        # Debug display
        self.show_debug = False
        
        # Create display window
        self.window_name = "SKU Content Display"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        if fullscreen:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.resizeWindow(self.window_name, resolution[0], resolution[1])
        
        logger.info(f"Video player initialized: {resolution[0]}x{resolution[1]} @ {fps}fps")
    
    def preload_video(self, video_id, video_path):
        """
        Preload a video into memory cache
        
        Args:
            video_id: Unique identifier for this video
            video_path: Path to video file
        """
        if not Path(video_path).exists():
            logger.warning(f"Video not found for preload: {video_path}")
            return False
        
        self.video_cache[video_id] = video_path
        logger.info(f"Preloaded video: {video_id} -> {video_path}")
        return True
    
    def open_video(self, video_path):
        """
        Open video file with hardware acceleration
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoCapture object or None if failed
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Open with hardware acceleration on Pi
        cap = cv2.VideoCapture(video_path, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Opened video: {video_path}")
            logger.info(f"  Resolution: {width}x{height}")
            logger.info(f"  FPS: {fps}")
            logger.info(f"  Frames: {frame_count}")
            
            return cap
        else:
            logger.error(f"Failed to open video: {video_path}")
            return None
    
    def read_frame(self, cap):
        """
        Read and preprocess frame from video
        
        Args:
            cap: VideoCapture object
            
        Returns:
            Processed frame or None
        """
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Resize to display resolution if needed
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def blend_frames(self, frame1, frame2, alpha):
        """
        Blend two frames with given alpha (fade)
        
        Args:
            frame1: First frame (fading out)
            frame2: Second frame (fading in)
            alpha: Blend factor (0.0 to 1.0)
            
        Returns:
            Blended frame
        """
        # Ensure both frames are same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Smooth fade using cosine interpolation (ease-in-out)
        smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
        
        # Blend frames
        blended = cv2.addWeighted(
            frame1, 1 - smooth_alpha,
            frame2, smooth_alpha,
            0
        )
        
        return blended
    
    def start(self, video_path):
        """
        Start playing initial video
        
        Args:
            video_path: Path to initial video
        """
        self.current_video = video_path
        self.current_cap = self.open_video(video_path)
        
        if self.current_cap is None:
            logger.error(f"Failed to start with video: {video_path}")
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        logger.info(f"Started playing: {video_path}")
    
    def switch_video(self, video_id, video_path):
        """
        Switch to a new video with fade transition
        
        Args:
            video_id: ID of video to switch to
            video_path: Path to video file
        """
        # Check if already playing this video
        if self.current_video == video_path and not self.transitioning:
            logger.debug(f"Already playing: {video_path}")
            return
        
        # Check if video exists
        if not Path(video_path).exists():
            logger.error(f"Cannot switch to non-existent video: {video_path}")
            return
        
        # Open next video
        self.next_video = video_path
        self.next_cap = self.open_video(video_path)
        
        if self.next_cap is None:
            logger.error(f"Failed to open next video: {video_path}")
            return
        
        # Start transition
        self.transitioning = True
        self.transition_frame = 0
        
        logger.info(f"Starting transition to: {video_path}")
    
    def update(self):
        """
        Update video player - read and display frame
        
        Returns:
            True if successful, False if should quit
        """
        if self.current_cap is None:
            return False
        
        # Read current frame
        current_frame = self.read_frame(self.current_cap)
        
        # Handle video loop
        if current_frame is None:
            # End of video - loop
            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = self.read_frame(self.current_cap)
            
            if current_frame is None:
                logger.error("Failed to loop video")
                return False
        
        # Handle transition
        if self.transitioning and self.next_cap is not None:
            # Read next frame
            next_frame = self.read_frame(self.next_cap)
            
            if next_frame is None:
                # Next video failed - abort transition
                logger.warning("Next video failed - aborting transition")
                self.next_cap.release()
                self.next_cap = None
                self.transitioning = False
            else:
                # Calculate fade alpha
                alpha = self.transition_frame / self.fade_frames
                alpha = min(1.0, max(0.0, alpha))
                
                # Blend frames
                display_frame = self.blend_frames(current_frame, next_frame, alpha)
                
                # Increment transition
                self.transition_frame += 1
                
                # Check if transition complete
                if self.transition_frame >= self.fade_frames:
                    # Complete transition
                    if self.current_cap:
                        self.current_cap.release()
                    
                    self.current_cap = self.next_cap
                    self.current_video = self.next_video
                    self.next_cap = None
                    self.next_video = None
                    self.transitioning = False
                    
                    logger.info(f"Transition complete: {self.current_video}")
        else:
            display_frame = current_frame
        
        # Add debug info if enabled
        if self.show_debug:
            display_frame = self.add_debug_overlay(display_frame)
        
        # Display frame
        cv2.imshow(self.window_name, display_frame)
        
        # Calculate FPS
        current_time = time.time()
        delta = current_time - self.last_frame_time
        if delta > 0:
            self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1.0 / delta)
        self.last_frame_time = current_time
        
        return True
    
    def add_debug_overlay(self, frame):
        """Add debug information overlay to frame"""
        overlay = frame.copy()
        
        # Debug info
        info = [
            f"FPS: {self.actual_fps:.1f}",
            f"Video: {Path(self.current_video).name if self.current_video else 'None'}",
            f"Transitioning: {self.transitioning}",
        ]
        
        if self.transitioning:
            progress = (self.transition_frame / self.fade_frames) * 100
            info.append(f"Transition: {progress:.0f}%")
        
        # Draw semi-transparent background
        overlay_height = len(info) * 30 + 20
        cv2.rectangle(overlay, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        y = 35
        for text in info:
            cv2.putText(
                frame, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )
            y += 30
        
        return frame
    
    def stop(self):
        """Stop video player and cleanup"""
        logger.info("Stopping video player...")
        
        if self.current_cap:
            self.current_cap.release()
        
        if self.next_cap:
            self.next_cap.release()
        
        cv2.destroyWindow(self.window_name)
        
        logger.info("Video player stopped")
