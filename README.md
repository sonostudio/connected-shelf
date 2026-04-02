# Connected Shelf

Real-time product detection with automatic content switching, built for Raspberry Pi 5 + OAK-D Pro.

Place a product in front of the camera → the display fades to its corresponding video. Remove it → back to default.

---

## Hardware Requirements

- Raspberry Pi 5 (8GB recommended)
- OAK-D Pro camera
- HDMI monitor
- Official Pi 5 power supply (27W)
- Active cooling (recommended)

---

## Prerequisites

Before starting, you'll need:

- A trained YOLOv8 model exported to `.blob` format — see [Model Training Guide](src/cv/README.md)
- Product videos in H.264 MP4 format (see [Video Preparation](#video-preparation) below)

---

## Installation

### 1. Install Raspberry Pi OS

Flash **Raspberry Pi OS 64-bit (Desktop)** using [Raspberry Pi Imager](https://www.raspberrypi.com/software/). Enable SSH during setup if you want headless access.

### 2. Install Python 3.11

Raspberry Pi OS ships with Python 3.13 by default, but this project requires 3.11. Build it from source:

```bash
sudo apt install -y build-essential libssl-dev zlib1g-dev \
  libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
  libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
  libffi-dev uuid-dev

wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
tar -xf Python-3.11.9.tgz
cd Python-3.11.9
./configure --enable-optimizations
make -j4
sudo make altinstall
cd ..
```

Verify the installation:

```bash
python3.11 --version
```

### 3. System dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-pip ffmpeg
```

### 4. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 5. Clone and install

```bash
git clone https://github.com/sonostudio/connected-shelf.git
cd connected-shelf
poetry env use python3.11
poetry install
```

### 6. Set up OAK-D Pro

Connect the camera to the **USB 3.0 port (blue)**, then run the following steps.

Set udev rules so the device is accessible without root:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Install Luxonis system dependencies:

```bash
sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
```

Verify the connection:

```bash
poetry run python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

You should see your device listed. If not, try a different USB port or reboot.

---

## Project Structure

```
connected-shelf/
├── config/
│   └── config.yaml            # Configuration
├── src/
│   ├── detect_and_display.py  # Main application
│   ├── video_player.py        # Video playback module
│   └── cv/                    # Model training scripts
├── model/                     # blob file and labels
├── videos/                    # Your product videos (user-supplied)
└── pyproject.toml
```

---

## Configuration

Edit `config/config.yaml` to map your SKU labels to video files:

```yaml
content:
  default: "videos/default.mp4"
  skus:
    product_a: "videos/product_a.mp4"
    product_b: "videos/product_b.mp4"
```

**SKU names must match exactly with the labels in your trained model.**

Other key settings:

```yaml
detection:
  model_path: "models/model.blob"
  labels_path: "models/labels.txt"
  confidence_threshold: 0.80   # Lower (e.g. 0.5) if detections are missed
  camera_fps: 30

display:
  resolution: [1920, 1080]
  fullscreen: true

transitions:
  fade_duration: 0.5           # Seconds
```

---

## Video Preparation

Videos must be H.264 MP4. Convert with FFmpeg:

```bash
ffmpeg -i input.mov -c:v libx264 -crf 20 -r 30 -s 1920x1080 -an output.mp4
```

Place converted files in the `videos/` directory.

---

## Running

```bash
poetry run python3 src/detect_and_display.py
```

**Controls:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset to default video |
| `d` | Toggle debug overlay |

---

## Model Training

See [src/cv/README.md](src/cv/README.md) for the full guide on training and exporting your YOLOv8 model.