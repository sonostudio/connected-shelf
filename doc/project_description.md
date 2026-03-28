## connected-shelf: Vision-Triggered Retail Display System

### Project Overview

connected-shelf is a real-time product detection system for retail environments. It uses an OAK-D Pro depth camera and a custom-trained YOLOv8 model to identify products placed in front of a display, then seamlessly fades to the corresponding product video — creating an interactive shelf that responds to physical product handling without any buttons, RFID tags, or embedded hardware.

* **Project type:** Retail technology prototype / interactive system
* **Intended audience:** Retail brands, trade show operators, product showroom designers

---

### Goal & Intent

The project was initiated to explore whether computer vision could make retail digital displays genuinely interactive — responding to what a customer picks up or places in front of a screen, rather than running on a predetermined schedule.

A central question guiding the work was how to build a system that is:

* **Frictionless** — no scanning, tapping, or explicit customer action required
* **Reliable in real conditions** — stable under varied lighting, backgrounds, and camera angles
* **Deployable on affordable edge hardware** — no cloud dependency, no recurring inference costs
* **Configurable without engineering** — new SKUs and videos mapped through a YAML file

The object, in this framing, becomes the interface. Placing a product in front of the screen is the interaction.

---

### Process

#### Concept & Feasibility Testing

Development began with object detection prototyping to validate whether SKU-level recognition was feasible for the target product category. Initial testing focused on detection stability, confidence levels under different lighting conditions, and camera positioning — prioritising reliability over model sophistication.

#### Dataset Preparation

Training data was captured on-site using the OAK-D Pro's RGB stream, recorded with a custom Python capture script. Videos were uploaded to Roboflow for frame extraction and annotation — drawing bounding boxes around each product and assigning SKU labels. Dataset augmentations (rotation, brightness, blur) were applied in Roboflow to improve generalisation to real-world variation.

#### Model Training

A YOLOv8n (nano) model was trained using Ultralytics on the custom dataset. The nano variant was chosen deliberately — it is fast enough for 30 FPS inference on Raspberry Pi 5 and small enough to convert to the MyriadX VPU's `.blob` format within the OAK-D Pro. Training used `imgsz=320` to match the on-device inference resolution.

#### Model Conversion & Deployment

The trained PyTorch model was exported to ONNX and then converted to a `.blob` file using the Luxonis online converter (tools.luxonis.com), targeting the OAK-D Pro's MyriadX VPU for on-device inference. This eliminates latency from sending frames to the host and keeps the Raspberry Pi free for video playback.

#### Video Playback System

A custom `VideoPlayer` class handles looping video playback with smooth fade transitions between SKU-specific content and a default idle video. Transitions use cosine interpolation for an ease-in-out feel. Videos are preloaded at startup for instant switching, and the player is designed to run concurrently with the detection loop via threading.

#### System Integration

The main application loop runs detection and video playback in separate threads — detection results trigger `switch_video()` calls on the player, which initiates a fade transition. A configurable return-to-default delay handles the case where a product is removed.

---

### Challenges & Learnings

#### Achieving stable confidence above 90%

Early model versions produced frequent false positives and missed detections. Iterating on dataset quality — more varied capture angles, better lighting coverage, and stricter annotation — proved more impactful than changing model architecture. The key insight was that production-ready detection accuracy comes from data quality, not algorithm novelty.

#### Blob conversion compatibility

Converting YOLOv8 ONNX models to DepthAI `.blob` format required specific ONNX opset and export settings to produce a compatible file. Using the Luxonis online converter with explicitly set parameters (FP16, 6 shaves, OpenVINO 2022.1) resolved compatibility issues that arose from local conversion attempts.

#### Concurrent detection and playback

Running detection and video playback in the same process required careful thread management to avoid frame drops in the video output. The detection loop runs as a daemon thread, with a shared lock protecting the state variable that controls which video is playing. This keeps the main thread focused on display performance.

#### Choosing the right model size for the hardware

Larger YOLOv8 variants (s, m) produced slightly better accuracy but were too slow for the MyriadX VPU at 30 FPS. The nano variant, combined with `imgsz=320`, gave a good balance of speed and accuracy for a small, controlled product catalogue.

---

### Output

#### Final system

* Raspberry Pi 5 application with DepthAI-based YOLOv8 inference pipeline
* Custom `VideoPlayer` with smooth fade transitions (cosine ease-in-out)
* YAML-based SKU-to-video configuration — no code changes needed to add new products
* End-to-end model training pipeline: capture → Roboflow annotation → YOLOv8 training → blob conversion → deployment
* On-site operator tools: camera test script, setup verification, video player test suite

#### User / customer experience

For a customer, the interaction is wordless and immediate. Picking up or placing a product in front of the display causes the screen to smoothly transition from ambient content to a video specific to that product. Removing the product fades it back. There is no visible interface — the product itself activates the media.

#### Media

* Video documentation: *to be added*

---

### Technical / Architecture Description

#### System overview

An OAK-D Pro camera runs YOLOv8n inference on-device via its MyriadX VPU. Detected SKU labels are passed to the host application on Raspberry Pi 5, which maps them to video files via a config file and triggers fade transitions in the video player.

#### Data flow

1. OAK-D Pro captures RGB frame at 30 FPS
2. MyriadX VPU runs YOLOv8n inference on-device
3. Detection results (label, confidence, bounding box) sent to Raspberry Pi 5 via USB
4. Host application filters by confidence threshold, extracts highest-confidence SKU label
5. SKU label looked up in YAML config to find video path
6. If SKU changes, `VideoPlayer.switch_video()` initiates fade transition
7. Video player blends current and next frame using cosine interpolation
8. Display output rendered to HDMI monitor

```
┌───────────────────────────────────────┐
│           Product Placement           │
│                                       │
│   Physical Item on Monitor Surface    │
│                                       │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│           OAK-D Pro Camera            │
│   RGB Capture + On-Device Inference   │
│   ┌───────────────────────────────┐   │
│   │  MyriadX VPU                  │   │
│   │  YOLOv8n (.blob)              │   │
│   └───────────────────────────────┘   │
└──────────────────┬────────────────────┘
                   │  USB
                   ▼
┌───────────────────────────────────────┐
│        Raspberry Pi 5                 │
│                                       │
│  Detection loop (daemon thread)       │
│  · Confidence filtering               │
│  · SKU → video mapping (YAML)         │
│  · State change detection             │
│                                       │
│  Video player (main thread)           │
│  · Looping playback                   │
│  · Cosine fade transitions            │
│  · Video preloading                   │
└──────────────────┬────────────────────┘
                   │  HDMI
                   ▼
┌───────────────────────────────────────┐
│          Display                      │
│  Default video / SKU-specific video   │
└───────────────────────────────────────┘
```

**Technologies**

* Hardware: Raspberry Pi 5, OAK-D Pro (MyriadX VPU)
* Detection: YOLOv8n (Ultralytics), DepthAI, Roboflow
* Video: OpenCV VideoCapture / VideoWriter
* Configuration: YAML
* Language: Python

**GitHub**

https://github.com/sonostudio/connected-shelf

---

### Technology Reusability & Other Use Cases

#### Reusable components

* On-device YOLOv8 inference pipeline for OAK-D Pro
* YAML-driven SKU-to-content mapping system
* VideoPlayer with smooth fade transitions (reusable for any triggered video display)
* End-to-end model training and deployment pipeline (capture → annotate → train → convert → deploy)

#### Alternative applications

##### Trade show and product demo environments

The same system can power interactive product demonstrations at trade shows or in showrooms — where picking up a product item triggers a video explanation, specification overview, or brand story without requiring staff intervention.

##### Museum and gallery interactives

Object recognition can trigger contextual content when a visitor picks up or examines a physical artefact — providing information, audio, or animation tied to specific objects rather than location alone.

##### Inventory and planogram monitoring

With a wider-angle camera and multi-object detection, the same pipeline can monitor shelf stock levels in real time — detecting gaps, misplaced items, or planogram deviations and triggering alerts.

#### Client value

connected-shelf demonstrates how computer vision can close the gap between physical product handling and digital content — making retail displays responsive to customer behaviour without RFID infrastructure, embedded hardware, or manual triggering. The edge-based architecture keeps operating costs low and latency imperceptible, while the configuration-driven design allows non-technical staff to add or update content mappings independently.