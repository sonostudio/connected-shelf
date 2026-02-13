## Interactive Retail Shelf

**Retail Technology Pilot**

An interactive retail display system that detects physical products in real time and dynamically plays mapped video content to create responsive, context-aware in-store media experiences.

Designed as a modular retail media infrastructure layer, the system bridges computer vision, edge deployment, and dynamic content logic — enabling seamless physical-digital interaction without RFID or embedded hardware.

---

## Goal & Intent

This project began as an exploration into cost-effective, computer-vision-driven retail media.

**Core question:**
How can digital narratives be introduced into physical retail environments in a way that is interactive, context-aware, and frictionless?

Unlike touchscreens or QR systems, customers do not need to scan, navigate, or search. Interaction occurs naturally through product handling.

By eliminating interface friction, the display becomes responsive rather than navigational.

The system enables SKU-level personalization and can expand to include:

* Product descriptions
* Styling recommendations
* Inventory information
* Promotional logic
* Interaction analytics

This shifts retail media from scheduled broadcasting to real-time, product-driven engagement.

---

## Process

### Concept & System Design

Development began with object detection prototyping to validate SKU-level recognition feasibility.

A custom-trained object detection model was selected to allow visual recognition without modifying physical products. Environmental testing included lighting variation, camera positioning, and background robustness to ensure stable real-time detection.

---

### Detection Pipeline Development

A custom YOLO model was trained in-house using Ultralytics and optimized for deployment on Luxonis OAK-D Pro hardware.

Pilot configuration:

* 5 SKUs
* On-device inference via DepthAI
* Class labels passed to playback system

Inference runs fully locally, ensuring low latency and deployment reliability.

Future expansion can incorporate bounding box coordinates to enable spatial overlays or generative media systems.

---

### Video Mapping & Playback Logic

Each SKU is mapped to video content via a configuration file.

Prototype behavior:

* Single-object interaction model
* 0.5-second fallback to default
* Multithreaded detection and playback
* Planned debounce and object-lock logic for production

The system prioritizes stability and clarity over rapid switching.

---

### Hardware Integration

* Raspberry Pi 5 (application & playback)
* Luxonis OAK-D Pro (on-device AI inference)
* HDMI display (horizontal placement)

All processing is edge-based. No cloud dependency is required.

---

## Output

When idle, the display plays branded default content with subtle placement guidance.

When a product is placed on the monitor surface, the system detects the SKU in real time and smoothly fades into product-specific video.

The interaction feels playful yet informative — the product itself appears to activate the media.

When removed, the system fades back to default within 0.5 seconds.

No buttons. No scanning. No navigation.
The object becomes the interface.

---

## Challenges & Learnings

Achieving stable model accuracy (>90% confidence) required careful dataset preparation, augmentation, and preprocessing.

Integrating custom YOLO models into the DepthAI framework required multiple iterations to produce compatible blob files.

A key insight from this project:

Production-ready systems require balance.
Rather than using the latest computationally heavy models, we optimized for hardware compatibility and real-world reliability.

Stability, efficiency, and deployment realism outweigh algorithm novelty in retail environments.

---

## Architecture Overview

```
┌──────────────────────────────────────┐
│           PRODUCT PLACEMENT          │
│                                      │
│  Physical Item on Monitor Surface    │
│                                      │
└─────────────────────┬────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│           OAK-D Pro Camera           │
│      (RGB Capture + On-Device ML)    │
└─────────────────────┬────────────────┘
                      │  USB
                      ▼
┌────────────────────────────────────────────────┐
│          Raspberry Pi 5 (Edge Node)            │
│                                                │
│  [Python Application]                          │
│  • Object detection (custom YOLO model)        │
│  • Inference via DepthAI pipeline              │
│  • Detection state logic                       │
│  • SKU result dispatch                         │
│                                                │
│  Output: SKU labels                            │
└─────────────────────┬──────────────────────────┘
                      │  Internal application logic
                      ▼
┌────────────────────────────────────────────────┐
│       Playback Logic (Python / OpenCV)         │
│                                                │
│  • SKU → Video mapping (config file)           │
│  • Responsive transition logic (fade)          │
│  • Default ↔ SKU specific state                │
│  • Debounce / object lock (future)             │
│                                                │
│  Output: Local video playback                  │
└────────────────────────────────────────────────┘
```

**Hardware**

* Raspberry Pi 5
* Luxonis OAK-D Pro
* HDMI Display

**Software**

* Python
* DepthAI
* Ultralytics YOLO
* OpenCV

**Deployment**

* Fully edge-based
* Multithreaded processing
* Config-driven SKU mapping
* No internet dependency

GitHub Repository:
[https://github.com/sonostudio/connected-shelf/tree/main](https://github.com/sonostudio/connected-shelf/tree/main)

---

## Reusability & Applications

Although developed for retail, the system functions as a modular vision-triggered interaction framework.

Applicable contexts include:

* Retail media networks
* Trade show installations
* Museum storytelling systems
* Amusement park interactive mechanics
* Product demo environments

The studio provides end-to-end implementation — from detection model training to system design and media integration — enabling seamless physical-digital blending.
