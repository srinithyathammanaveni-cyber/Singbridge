# SignBridge v2.0 🤟

**ASL Sign Language ↔ Text & Speech**  
MediaPipe hand detection · Node.js backend · Real-time 2-hand recognition

---

## Why camera was denied before

Browsers block camera access on `file://` URLs.  
You MUST open the app via **http://localhost:5000** (served by Node.js).

---

## Folder Structure

```
signbridge/
├── server.js               ← Node.js backend (Express)
├── recognize_worker.py     ← Python/MediaPipe worker (persistent)
├── gesture_model.py        ← ASL classifier (2-hand support)
├── preprocessing.py        ← Image preprocessing pipeline
├── package.json
├── README.md
├── models/                 ← hand_landmarker.task (auto-downloads ~10MB)
└── public/
    └── index.html          ← Full frontend UI (3 modes)
```

---

## Setup (3 steps)

### Step 1 — Install Node.js dependencies
```bash
npm install
```

### Step 2 — Install Python dependencies
```bash
pip install mediapipe opencv-python numpy
```

### Step 3 — Start the server
```bash
node server.js
```

Then open your browser at: **http://localhost:5000**

> ✅ Camera will work because we're on localhost  
> ✅ MediaPipe model downloads automatically on first run (~10MB)  
> ✅ MongoDB is optional — app works without it

---

## Features

### Mode 1: Sign → Text
- Real-time webcam capture
- MediaPipe detects **up to 2 hands** simultaneously
- Skeleton overlay drawn on video
- Gesture confirmed after 6 stable frames (prevents jitter)
- Auto-builds a sentence from confirmed signs
- **Text-to-speech** readback
- History panel — click any sign to add to sentence

### Mode 2: Text → Sign Animation
- Type any word or sentence
- Animated hand skeleton shows each letter in ASL
- Adjustable speed (0.3s – 2s per letter)
- Quick phrase buttons
- Letter queue shows progress

### Mode 3: Reference
- Visual guide to all supported signs
- Hand skeleton diagram for each gesture

---

## Supported Signs

| Gesture | Triggered by |
|---------|-------------|
| A | Fist |
| B / 4 | Four fingers up |
| C | Curved hand |
| D / 1 | Index finger only |
| Good 👍 | Thumbs up |
| Hello / B | Open hand |
| I | Pinky only |
| I Love You | Thumb + index + pinky |
| K / P | Thumb + index + middle |
| L | Thumb + index (L-shape) |
| O | Pinched circle |
| U / H | Two fingers together |
| V / 2 ✌ | Peace sign |
| W / 3 | Three fingers |
| Y | Thumb + pinky |
| Hello (2-hand) | Both hands open |
| Stop (2-hand) | Both fists |
| Very Good | Both thumbs up |
| I Love You x2 | Both ILY signs |

---

## Troubleshooting

**Camera access denied:**  
→ Make sure you opened `http://localhost:5000` NOT `file://...`

**Python worker not starting:**  
→ Set `PYTHON_PATH` environment variable: `PYTHON_PATH=C:\Python311\python.exe node server.js`  
→ Or on Mac/Linux: `PYTHON_PATH=/usr/bin/python3 node server.js`

**MediaPipe not found:**  
→ Run: `pip install mediapipe`

**No gestures detected:**  
→ Ensure good lighting, hold hand 30-60cm from camera  
→ Check the server console for Python worker logs
