"""
play_mac.py — Mac side
-----------------------
Listens for OSC messages from the Pi and drives video playback.

OSC messages handled:
  /detection   sku_label(str)  confidence(float)  x1 y1 x2 y2 (floats)
  /no_detection  (no args)

Run from project root:
    poetry run python src/play_mac.py
"""

import time
import logging
import threading
import yaml
import cv2
from pathlib import Path
from pythonosc import dispatcher, osc_server
from video_player import VideoPlayer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Mac] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# OSC state — written by the OSC thread, read by the main video loop
# ---------------------------------------------------------------------------

class DetectionState:
    """Thread-safe container for the latest detection from the Pi."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sku        = None   # str | None
        self._confidence = 0.0
        self._bbox       = None   # (x1, y1, x2, y2) | None
        self._last_msg   = 0.0   # epoch time of last OSC message

    # -- writers (OSC thread) ------------------------------------------------

    def set_detection(self, sku, confidence, x1, y1, x2, y2):
        with self._lock:
            self._sku        = sku
            self._confidence = confidence
            self._bbox       = (x1, y1, x2, y2)
            self._last_msg   = time.time()

    def set_no_detection(self):
        with self._lock:
            self._sku        = None
            self._confidence = 0.0
            self._bbox       = None
            self._last_msg   = time.time()

    # -- readers (main thread) -----------------------------------------------

    def get(self):
        """Returns (sku, confidence, bbox, last_msg_time)."""
        with self._lock:
            return self._sku, self._confidence, self._bbox, self._last_msg


# ---------------------------------------------------------------------------
# OSC handlers
# ---------------------------------------------------------------------------

def make_handlers(state: DetectionState):
    def on_detection(address, sku, confidence, x1, y1, x2, y2):
        logger.info(
            f"← /detection  {sku}  {confidence:.2f}"
            f"  [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]"
        )
        state.set_detection(sku, confidence, x1, y1, x2, y2)

    def on_no_detection(address):
        logger.debug("← /no_detection")
        # Intentionally ignored — we rely on no_detection_timeout instead.
        # Reacting immediately causes flickering when confidence briefly dips.

    return on_detection, on_no_detection


def start_osc_server(state, listen_ip, port):
    """Start the OSC server in a daemon thread."""
    on_detection, on_no_detection = make_handlers(state)

    d = dispatcher.Dispatcher()
    d.map("/detection",    on_detection)
    d.map("/no_detection", on_no_detection)

    server = osc_server.ThreadingOSCUDPServer((listen_ip, port), d)
    logger.info(f"OSC server listening on {listen_ip}:{port}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


# ---------------------------------------------------------------------------
# Main video loop
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    disp_cfg = cfg["display"]
    trans_cfg = cfg["transitions"]
    osc_cfg   = cfg["osc"]
    content   = cfg["content"]

    no_detection_timeout = osc_cfg["no_detection_timeout"]
    # ADDED: confidence gate — detections below this are ignored on the Mac side.
    conf_threshold       = cfg["detection"]["confidence_threshold"]
    sku_map              = content["skus"]          # {label: video_path}
    default_video        = content["default"]

    # Validate videos
    missing = []
    for path in [default_video] + list(sku_map.values()):
        if not Path(path).exists():
            missing.append(path)
    if missing:
        for p in missing:
            logger.warning(f"Video not found: {p}")

    # OSC state
    state = DetectionState()

    # OSC server — listen on all interfaces so the Pi can reach us
    start_osc_server(state, "0.0.0.0", osc_cfg["port"])

    # Video player
    player = VideoPlayer(
        resolution=tuple(disp_cfg["resolution"]),
        fps=disp_cfg["fps"],
        fullscreen=disp_cfg["fullscreen"],
        fade_duration=trans_cfg["fade_duration"],
    )

    if cfg["video"]["preload"]:
        logger.info("Preloading videos…")
        player.preload_video("default", default_video)
        for sku, path in sku_map.items():
            player.preload_video(sku, path)

    player.start(default_video)
    current_sku = None   # tracks what is currently playing

    logger.info("Playback loop running. Press 'q' to quit, 'r' to reset.")

    frame_interval = 1.0 / disp_cfg["fps"]
    next_frame_time = time.time()

    try:
        while True:
            # ---------- read latest detection state -------------------------
            sku, confidence, bbox, last_msg = state.get()

            # If no OSC message has arrived recently, treat as no-detection
            timed_out = (
                last_msg > 0 and
                (time.time() - last_msg) > no_detection_timeout
            )
            if timed_out:
                sku = None

            # ADDED: confidence gate — ignore detections below threshold.
            # The Pi already filtered with pre_filter_threshold (noise floor);
            # this is the real threshold that controls video switching.
            if sku is not None and confidence < conf_threshold:
                logger.debug(
                    f"  {sku} {confidence:.2%} below confidence_threshold "
                    f"({conf_threshold:.2%}) — ignoring"
                )
                sku = None

            # ---------- react to SKU changes --------------------------------
            if sku != current_sku:
                if sku is None:
                    logger.info("→ No detection — switching to default video")
                    player.switch_video("default", default_video)
                elif sku in sku_map:
                    logger.info(f"→ Detected '{sku}' ({confidence:.2%}) — switching video")
                    player.switch_video(sku, sku_map[sku])
                else:
                    logger.warning(f"→ Unknown SKU '{sku}' — switching to default")
                    player.switch_video("default", default_video)
                current_sku = sku

            # ---------- update display --------------------------------------
            if not player.update():
                break

            # ---------- keyboard controls -----------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested.")
                break
            elif key == ord("r"):
                logger.info("Manual reset to default.")
                state.set_no_detection()
            elif key == ord("d"):
                player.show_debug = not player.show_debug

            # ---------- frame pacing ----------------------------------------
            # VideoPlayer.update() doesn't sleep; we pace here so we don't
            # spin the CPU harder than needed.
            now = time.time()
            sleep_for = next_frame_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_frame_time = max(now, next_frame_time) + frame_interval

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        player.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()