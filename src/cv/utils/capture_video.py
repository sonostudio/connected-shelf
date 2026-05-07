import cv2
import depthai as dai
import os
import subprocess
import warnings
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time


@dataclass
class CaptureConfig:
    dataset_dir: Path = Path("../data/raw/")
    label_id: str = "0"

    # Camera settings
    preview_size: tuple[int, int] = (640, 640)  # Match YOLO training size
    fps: int = 30

    # Video codec (mp4v = software MPEG-4, reliable on Pi 5)
    # Output is post-processed to H.264 via ffmpeg for browser compatibility
    fourcc: str = 'mp4v'

    # Enhanced vision settings
    contrast: int = 5
    sharpness: int = 5
    saturation: int = 0


class VideoRecorder:
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.recording: bool = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_filename: Optional[str] = None
        self.frame_count: int = 0
        self.start_time: Optional[float] = None

    def start(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = f"{self.config.label_id}_{timestamp}.mp4"
        filepath = self.config.dataset_dir / self.current_filename

        fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
        self.video_writer = cv2.VideoWriter(
            str(filepath),
            fourcc,
            float(self.config.fps),
            self.config.preview_size
        )

        if not self.video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {filepath}")

        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        print(f"Recording started: {self.current_filename}")

    def write_frame(self, frame) -> None:
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
            self.frame_count += 1

    def stop(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.recording:
            duration = time.time() - self.start_time if self.start_time else 0
            print(f"Recording saved: {self.current_filename}")
            print(f"Frames: {self.frame_count}, Duration: {duration:.1f}s")
            self._convert_to_h264()

        self.recording = False
        self.frame_count = 0
        self.start_time = None

    def _convert_to_h264(self) -> None:
        if self.current_filename is None:
            return

        src = self.config.dataset_dir / self.current_filename
        tmp = src.with_suffix(".h264.mp4")

        print(f"Converting to H.264 (browser-compatible)...")
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-vcodec", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                str(tmp),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"ERROR: ffmpeg conversion failed:")
            print(result.stderr)
            return

        # Replace original mp4v file with H.264 version
        src.unlink()
        tmp.rename(src)
        print(f"Conversion complete: {self.current_filename}")

    def get_stats(self) -> tuple[int, float]:
        if not self.recording or self.start_time is None:
            return 0, 0.0
        return self.frame_count, time.time() - self.start_time

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_pipeline(config: CaptureConfig) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    # Define Color Camera
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(*config.preview_size)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setFps(config.fps)
    # Scale 1080p -> 720p (2/3), giving enough resolution for a 640x640 preview crop
    cam.setIspScale(2, 3)

    # Enhanced vision settings
    cam.initialControl.setContrast(config.contrast)
    cam.initialControl.setSharpness(config.sharpness)
    cam.initialControl.setSaturation(config.saturation)

    # XLinkOut (Output to host)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    return pipeline


def add_overlay(frame, recording: bool, frame_count: int = 0, duration: float = 0.0):
    display_frame = frame.copy()

    if recording:
        cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(display_frame, "REC", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        stats_text = f"Frames: {frame_count} | Time: {duration:.1f}s"
        cv2.putText(display_frame, stats_text, (10, 630),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return display_frame


def main():
    # --- CONFIG ---
    config = CaptureConfig(
        label_id="0"  # ← CHANGE THIS BETWEEN SESSIONS
    )

    try:
        config.dataset_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Cannot create dataset directory: {e}")
        return

    print(f"OAK-D Video Recorder initialized.")
    print(f"Target Label ID: '{config.label_id}'")
    print(f"Output Directory: {config.dataset_dir}")
    print(f"⌨Controls: [R] to Start/Stop Recording, [Q] to Quit")

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    pipeline = create_pipeline(config)

    try:
        with dai.Device(pipeline) as device:
            print("OAK-D camera connected")

            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            with VideoRecorder(config) as recorder:
                while True:
                    in_rgb = q_rgb.tryGet()

                    if in_rgb is not None:
                        frame = in_rgb.getCvFrame()
                        recorder.write_frame(frame)

                        frame_count, duration = recorder.get_stats()
                        display_frame = add_overlay(
                            frame,
                            recorder.recording,
                            frame_count,
                            duration
                        )
                        cv2.imshow("Data Capture - OAK-D", display_frame)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print("Shutting down...")
                        break
                    elif key == ord('r'):
                        if not recorder.recording:
                            try:
                                recorder.start()
                            except RuntimeError as e:
                                print(f"ERROR: Failed to start recording: {e}")
                                print("Starting over...")
                        else:
                            recorder.stop()

    except RuntimeError as e:
        print(f"ERROR: OAK-D camera not connected or failed to initialize")
        print(f"   Details: {e}")
        return

    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return

    finally:
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()