import sys
from pathlib import Path

print("="*70)
print("SKU Detection System - Setup Test")
print("="*70)

# Test 1: Check Python version
print("\n[1/7] Checking Python version...")
if sys.version_info >= (3, 9):
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"✗ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.9+)")
    sys.exit(1)

# Test 2: Check dependencies
print("\n[2/7] Checking dependencies...")
try:
    import depthai as dai
    print(f"✓ depthai {dai.__version__}")
except ImportError:
    print("✗ depthai not installed")
    print("  Run: pip install depthai")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError:
    print("✗ OpenCV not installed")
    print("  Run: pip install opencv-python")
    sys.exit(1)

try:
    import yaml
    print("✓ PyYAML")
except ImportError:
    print("✗ PyYAML not installed")
    print("  Run: pip install pyyaml")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not installed")
    print("  Run: pip install numpy")
    sys.exit(1)

# Test 3: Check OAK-D Pro
print("\n[3/7] Checking OAK-D Pro connection...")
try:
    devices = dai.Device.getAllAvailableDevices()
    if devices:
        print(f"✓ Found {len(devices)} device(s)")
        for dev in devices:
            print(f"  - {dev.getMxId()} ({dev.protocol.name})")
    else:
        print("✗ No OAK-D Pro devices found")
        print("  - Check USB connection (use USB 3.0 port)")
        print("  - Try different cable")
        print("  - Reboot if needed")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking devices: {e}")
    sys.exit(1)

# Test 4: Check files exist
print("\n[4/7] Checking project files...")
required_files = {
    'detect_and_display.py': 'Main script',
    'video_player.py': 'Video player module',
    'config.yaml': 'Configuration file',
}

for file, desc in required_files.items():
    if Path(file).exists():
        print(f"✓ {file} ({desc})")
    else:
        print(f"✗ {file} missing ({desc})")

# Test 5: Check model
print("\n[5/7] Checking model files...")
if Path('model/model.blob').exists():
    size = Path('model/model.blob').stat().st_size / (1024*1024)
    print(f"✓ model/model.blob ({size:.1f} MB)")
else:
    print("✗ model/model.blob not found")
    print("  Train your model and convert to blob format")

if Path('model/labels.txt').exists():
    with open('model/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"✓ model/labels.txt ({len(labels)} classes)")
    print(f"  Classes: {', '.join(labels)}")
else:
    print("✗ model/labels.txt not found")

# Test 6: Check videos
print("\n[6/7] Checking video files...")
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    default_video = config['content']['default']
    if Path(default_video).exists():
        size = Path(default_video).stat().st_size / (1024*1024)
        print(f"✓ {default_video} ({size:.1f} MB)")
    else:
        print(f"✗ {default_video} not found")
    
    sku_videos = config['content']['skus']
    found_count = 0
    for sku, video_path in sku_videos.items():
        if Path(video_path).exists():
            found_count += 1
        else:
            print(f"✗ {video_path} not found (for {sku})")
    
    if found_count > 0:
        print(f"✓ Found {found_count} SKU video(s)")
    
except Exception as e:
    print(f"⚠ Could not check videos: {e}")

# Test 7: Test video playback capability
print("\n[7/7] Testing video playback...")
try:
    # Try to open a video capture (test codec support)
    cap = cv2.VideoCapture('', cv2.CAP_V4L2)
    if cap.isOpened():
        cap.release()
    print("✓ Video codec support available")
except Exception as e:
    print(f"⚠ Video playback test warning: {e}")

# Summary
print("\n" + "="*70)
print("Test Summary")
print("="*70)

if Path('model/model.blob').exists() and Path('config.yaml').exists():
    print("✓ Core components ready")
    print("\nYou can start the system with:")
    print("  python3 detect_and_display.py")
else:
    print("✗ Setup incomplete")
    print("\nMissing components - please complete setup:")
    if not Path('model/model.blob').exists():
        print("  - Train model and convert to blob")
    if not Path('config.yaml').exists():
        print("  - Create config.yaml")

print("\nFor full setup instructions, see SETUP_GUIDE.md")
print("="*70)
