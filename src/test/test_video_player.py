"""
Video Player Test Script
Test video playback and transitions without requiring OAK-D Pro or model

This script lets you:
- Test video playback performance
- Test fade transitions
- Try different fade durations
- Switch between videos manually
- Monitor FPS and performance
"""

import cv2
import time
import sys
import threading
from pathlib import Path

# Add parent directory to path to import video_player
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_player import VideoPlayer

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Video files to test (paths relative to project root)
TEST_VIDEOS = {
    '1': 'videos/default.mp4',
    '2': 'videos/loop1.mov',
    '3': 'videos/loop2.mp4',
}

# Display settings
RESOLUTION = (1920, 1080)  # Change to (1280, 720) for 720p
FULLSCREEN = True
FPS = 30

# Transition settings
FADE_DURATION = 0.5  # seconds


# ============================================================================
# STDIN INPUT HELPER
# ============================================================================

class TerminalInput:
    """
    Reads input from terminal in a background thread so the video
    loop can keep running while waiting for commands.
    """
    def __init__(self):
        self.command = None
        self.lock = threading.Lock()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while True:
            try:
                line = sys.stdin.readline().strip()
                if line:
                    with self.lock:
                        self.command = line
            except EOFError:
                break

    def get_command(self):
        """Returns the latest command and clears it, or None if no input."""
        with self.lock:
            cmd = self.command
            self.command = None
            return cmd


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def get_project_root():
    """Get project root directory (parent of src/)"""
    # From src/test/test_video_player.py -> go up 2 levels to project root
    return Path(__file__).parent.parent.parent


def list_available_videos():
    """List all videos that exist"""
    project_root = get_project_root()
    print(f"\nProject root: {project_root}")
    print("Available videos:")
    available = {}
    for key, path in TEST_VIDEOS.items():
        full_path = project_root / path
        if full_path.exists():
            size = full_path.stat().st_size / (1024*1024)
            print(f"  [{key}] {path} ({size:.1f} MB)")
            available[key] = str(full_path)
        else:
            print(f"  [{key}] {path} - NOT FOUND")
            print(f"       (looking at: {full_path})")
    return available


def test_basic_playback(video_path, duration=10):
    """Test basic video playback without transitions"""
    print("\n" + "="*70)
    print(f"Test 1: Basic Playback ({duration} seconds)")
    print("="*70)
    print(f"Video: {video_path}")
    print("Testing smooth playback and looping...")

    if not Path(video_path).exists():
        print(f"✗ Video not found: {video_path}")
        return False

    player = VideoPlayer(
        resolution=RESOLUTION,
        fps=FPS,
        fullscreen=FULLSCREEN,
        fade_duration=FADE_DURATION
    )

    player.start(video_path)
    player.show_debug = True

    start_time = time.time()
    frame_count = 0

    terminal = TerminalInput()
    print("\nPlaying... (type 'q' + Enter to stop early)")

    while time.time() - start_time < duration:
        if not player.update():
            break

        frame_count += 1
        cv2.waitKey(1)

        cmd = terminal.get_command()
        if cmd == 'q':
            break

    player.stop()

    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0

    print(f"\n✓ Test complete!")
    print(f"  Frames: {frame_count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  FPS: {actual_fps:.1f}")

    if actual_fps >= FPS * 0.9:  # Within 90% of target
        print(f"  Performance: ✓ GOOD (target: {FPS} fps)")
        return True
    else:
        print(f"  Performance: ⚠ LOW (target: {FPS} fps)")
        print("  Consider: Lower resolution or reduce bitrate")
        return False


def test_transitions(video1_path, video2_path, fade_duration=0.5):
    """Test transitions between two videos"""
    print("\n" + "="*70)
    print(f"Test 2: Transition Test (fade: {fade_duration}s)")
    print("="*70)
    print(f"Video 1: {video1_path}")
    print(f"Video 2: {video2_path}")
    print(f"Fade duration: {fade_duration}s")

    if not Path(video1_path).exists() or not Path(video2_path).exists():
        print("✗ One or both videos not found")
        return False

    player = VideoPlayer(
        resolution=RESOLUTION,
        fps=FPS,
        fullscreen=FULLSCREEN,
        fade_duration=fade_duration
    )

    player.start(video1_path)
    player.show_debug = True

    print("\nSequence:")
    print("  1. Play video 1 for 5 seconds")
    print("  2. Transition to video 2")
    print("  3. Play video 2 for 5 seconds")
    print("  4. Transition back to video 1")
    print("\nType 'q' + Enter to quit anytime")

    terminal = TerminalInput()
    transitions_completed = 0

    def update_or_quit():
        """Returns True if should quit."""
        if not player.update():
            return True
        cv2.waitKey(1)
        return terminal.get_command() == 'q'

    # Play video 1
    print("\n[Playing video 1...]")
    start = time.time()
    while time.time() - start < 5:
        if update_or_quit():
            player.stop()
            return False

    # Transition to video 2
    print("[Transitioning to video 2...]")
    player.switch_video('video2', video2_path)

    transition_start = time.time()
    while player.transitioning:
        if update_or_quit():
            player.stop()
            return False
        if time.time() - transition_start > fade_duration * 2:
            break

    transitions_completed += 1
    print(f"  ✓ Transition 1 complete ({time.time() - transition_start:.2f}s)")

    # Play video 2
    print("[Playing video 2...]")
    start = time.time()
    while time.time() - start < 5:
        if update_or_quit():
            player.stop()
            return False

    # Transition back to video 1
    print("[Transitioning back to video 1...]")
    player.switch_video('video1', video1_path)

    transition_start = time.time()
    while player.transitioning:
        if update_or_quit():
            player.stop()
            return False
        if time.time() - transition_start > fade_duration * 2:
            break

    transitions_completed += 1
    print(f"  ✓ Transition 2 complete ({time.time() - transition_start:.2f}s)")

    # Play for a bit more
    start = time.time()
    while time.time() - start < 3:
        if update_or_quit():
            break

    player.stop()

    print(f"\n✓ Test complete!")
    print(f"  Transitions: {transitions_completed}/2")

    if transitions_completed == 2:
        print("  Result: ✓ Transitions working smoothly")
        return True
    else:
        print("  Result: ⚠ Transitions incomplete")
        return False


def interactive_test(available_videos):
    """Interactive mode - manually switch between videos"""
    print("\n" + "="*70)
    print("Test 3: Interactive Manual Testing")
    print("="*70)
    print("\nCommands (type + Enter):")
    print("  [1-9] - Switch to video number")
    print("  [d]   - Toggle debug info")
    print("  [+]   - Increase fade duration")
    print("  [-]   - Decrease fade duration")
    print("  [q]   - Quit")

    if not available_videos:
        print("\n✗ No videos available")
        return

    # Get first video
    first_key = list(available_videos.keys())[0]
    first_video = available_videos[first_key]

    current_fade = FADE_DURATION

    player = VideoPlayer(
        resolution=RESOLUTION,
        fps=FPS,
        fullscreen=FULLSCREEN,
        fade_duration=current_fade
    )

    player.start(first_video)
    player.show_debug = True
    current_video = first_key

    print(f"\nStarted with video [{current_video}]: {first_video}")
    print("\nTry switching between videos to test transitions!")

    terminal = TerminalInput()

    while True:
        if not player.update():
            break

        cv2.waitKey(1)

        cmd = terminal.get_command()
        if cmd is None:
            continue

        if cmd == 'q':
            break

        elif cmd == 'd':
            player.show_debug = not player.show_debug
            print(f"Debug: {'ON' if player.show_debug else 'OFF'}")

        elif cmd == '+':
            current_fade = min(2.0, current_fade + 0.1)
            player.fade_duration = current_fade
            player.fade_frames = int(player.fps * current_fade)
            print(f"Fade duration: {current_fade:.1f}s")

        elif cmd == '-':
            current_fade = max(0.1, current_fade - 0.1)
            player.fade_duration = current_fade
            player.fade_frames = int(player.fps * current_fade)
            print(f"Fade duration: {current_fade:.1f}s")

        elif cmd in available_videos:
            if cmd != current_video:
                video_path = available_videos[cmd]
                print(f"Switching to [{cmd}]: {video_path}")
                player.switch_video(cmd, video_path)
                current_video = cmd

    player.stop()
    print("\n✓ Interactive test ended")


def test_different_fade_durations(video1_path, video2_path):
    """Test multiple fade durations to find the best one"""
    print("\n" + "="*70)
    print("Test 4: Fade Duration Comparison")
    print("="*70)
    print("Testing different fade durations to find optimal setting...")

    if not Path(video1_path).exists() or not Path(video2_path).exists():
        print("✗ Videos not found")
        return

    durations = [0.2, 0.3, 0.5, 0.7, 1.0]

    print("\nYou will see transitions with different fade durations.")
    print("Pay attention to which feels most natural.\n")

    for duration in durations:
        print(f"\n--- Testing fade duration: {duration}s ---")
        input("Press Enter to continue...")

        player = VideoPlayer(
            resolution=RESOLUTION,
            fps=FPS,
            fullscreen=FULLSCREEN,
            fade_duration=duration
        )

        terminal = TerminalInput()

        # Start with video 1
        player.start(video1_path)

        # Play for 3 seconds
        start = time.time()
        while time.time() - start < 3:
            if not player.update():
                break
            cv2.waitKey(1)
            if terminal.get_command() == 'q':
                player.stop()
                return

        # Transition to video 2
        print(f"  [Transitioning with {duration}s fade...]")
        player.switch_video('video2', video2_path)

        # Wait for transition
        while player.transitioning:
            if not player.update():
                break
            cv2.waitKey(1)
            if terminal.get_command() == 'q':
                player.stop()
                return

        # Play for 2 seconds
        start = time.time()
        while time.time() - start < 2:
            if not player.update():
                break
            cv2.waitKey(1)
            if terminal.get_command() == 'q':
                player.stop()
                return

        player.stop()
        print(f"  ✓ Done")

    print("\n✓ Comparison complete!")
    print("\nRecommendations:")
    print("  • 0.3-0.5s: Quick, snappy transitions")
    print("  • 0.5-0.7s: Smooth, professional feel")
    print("  • 0.7-1.0s: Slow, gentle transitions")


# ============================================================================
# MAIN TEST MENU
# ============================================================================

def main():
    """Main test menu"""
    print("="*70)
    print("Video Player Test Script for Raspberry Pi")
    print("="*70)
    print("\nThis script tests video playback and transitions")
    print("WITHOUT requiring OAK-D Pro or trained model.")

    # List available videos
    available_videos = list_available_videos()

    if not available_videos:
        print("\n✗ No test videos found!")
        print("\nPlease add videos to the 'videos/' directory:")
        for key, path in TEST_VIDEOS.items():
            print(f"  {path}")
        print("\nOr edit TEST_VIDEOS in this script to point to your videos.")
        sys.exit(1)

    # Get test videos
    video_keys = list(available_videos.keys())
    video1 = available_videos[video_keys[0]]
    video2 = available_videos[video_keys[1]] if len(video_keys) > 1 else video1

    # Main menu
    while True:
        print("\n" + "="*70)
        print("Test Menu")
        print("="*70)
        print("\n1. Basic Playback Test (10 seconds)")
        print("2. Transition Test (automatic)")
        print("3. Interactive Manual Test")
        print("4. Compare Different Fade Durations")
        print("5. Quick Transition Test (rapid switching)")
        print("6. Settings")
        print("q. Quit")

        choice = input("\nSelect test (1-6, q): ").strip().lower()

        if choice == '1':
            test_basic_playback(video2, duration=10)

        elif choice == '2':
            test_transitions(video1, video2, FADE_DURATION)

        elif choice == '3':
            interactive_test(available_videos)

        elif choice == '4':
            test_different_fade_durations(video1, video2)

        elif choice == '5':
            print("\nQuick Transition Test")
            print("Rapidly switching between videos...")
            test_transitions(video1, video2, fade_duration=0.3)

        elif choice == '6':
            print("\nCurrent Settings:")
            print(f"  Resolution: {RESOLUTION[0]}x{RESOLUTION[1]}")
            print(f"  FPS: {FPS}")
            print(f"  Fullscreen: {FULLSCREEN}")
            print(f"  Fade Duration: {FADE_DURATION}s")
            print("\nEdit this script to change settings (see top of file)")

        elif choice == 'q':
            print("\nExiting...")
            break

        else:
            print("\n✗ Invalid choice")

    print("\n✓ Tests complete!")
    print("\nPerformance Tips:")
    print("  • If video stutters: Use 720p or lower bitrate")
    print("  • If transitions lag: Reduce fade_duration")
    print("  • If CPU high: Lower camera_fps in final system")
    print("  • Check temperature: vcgencmd measure_temp")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()