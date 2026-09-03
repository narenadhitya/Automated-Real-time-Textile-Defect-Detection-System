"""
backend_server.py
=================
FastAPI backend serving:
  - MJPEG video stream with defect annotations at /api/video-feed
  - REST APIs for class switching, stats, heatmap, pause, screenshot

Usage:
    pip install fastapi uvicorn
    python backend_server.py
    # Then open http://localhost:8000
"""

import json
import os
import sys
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_STREAM_URL = "http://10.245.211.49:8080/video"
MODEL_DIR = Path("textile_models")
SAVE_BASE = Path("saved_images")
LOG_PATH = Path("log.txt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_ORDER = [
    "class_1_fine_texture",
    "class_2_stochastic_texture",
    "class_3_periodic_texture",
    "class_4_printed_nonperiodic",
]

CLASS_DISPLAY_NAMES = {
    "class_1_fine_texture": "Fine Texture",
    "class_2_stochastic_texture": "Stochastic",
    "class_3_periodic_texture": "Periodic",
    "class_4_printed_nonperiodic": "Printed",
}

CLASS_DESCRIPTIONS = {
    "class_1_fine_texture": "Dense, fine weave patterns with uniform texture",
    "class_2_stochastic_texture": "Random, irregular surface patterns",
    "class_3_periodic_texture": "Regular repeating geometric patterns",
    "class_4_printed_nonperiodic": "Printed designs with complex, non-repeating motifs",
}

CLASS_COLORS_HEX = {
    "class_1_fine_texture": "#3AB4F2",
    "class_2_stochastic_texture": "#F2DC3A",
    "class_3_periodic_texture": "#3AF252",
    "class_4_printed_nonperiodic": "#C864FF",
}

CLASS_COLORS_BGR = {
    "class_1_fine_texture": (255, 180, 50),
    "class_2_stochastic_texture": (50, 220, 220),
    "class_3_periodic_texture": (50, 200, 50),
    "class_4_printed_nonperiodic": (200, 100, 255),
}

BACKBONE_NAMES = ["convnext_tiny", "convnext_small", "swin_t",
                  "efficientnet_v2_s", "efficientnet_b3"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ============================================================================
# MODEL BUILDING
# ============================================================================
def build_defect_model(backbone, dropout_feat=0.40, dropout_mid=0.20):
    if backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat), nn.Linear(in_f, 128), nn.GELU(),
            nn.Dropout(p=dropout_mid), nn.Linear(128, 1))
    elif backbone == "convnext_small":
        model = models.convnext_small(weights=None)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat), nn.Linear(in_f, 128), nn.GELU(),
            nn.Dropout(p=dropout_mid), nn.Linear(128, 1))
    elif backbone == "swin_t":
        model = models.swin_t(weights=None)
        in_f = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_feat), nn.Linear(in_f, 128), nn.GELU(),
            nn.Dropout(p=dropout_mid), nn.Linear(128, 1))
    elif backbone == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True), nn.Linear(in_f, 128),
            nn.SiLU(inplace=True), nn.Dropout(p=dropout_mid), nn.Linear(128, 1))
    elif backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True), nn.Linear(in_f, 128),
            nn.SiLU(inplace=True), nn.Dropout(p=dropout_mid), nn.Linear(128, 1))
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


def load_defect_model(model_dir, class_name, device):
    model_path = None
    meta_path = None
    for bb in BACKBONE_NAMES:
        mp = model_dir / f"{class_name}_highacc_{bb}.pth"
        mt = model_dir / f"{class_name}_highacc_{bb}_meta.json"
        if mp.exists():
            model_path = mp
            meta_path = mt
            break

    if model_path is None:
        raise FileNotFoundError(f"No model for {class_name}")

    meta = {"backbone": "convnext_tiny", "image_size": 288,
            "decision_threshold": 0.5, "dropout_feat": 0.45, "dropout_mid": 0.20}
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta.update(json.load(f))

    model = build_defect_model(meta["backbone"],
                               meta.get("dropout_feat", 0.45),
                               meta.get("dropout_mid", 0.20))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, meta


# ============================================================================
# PREPROCESSING / INFERENCE
# ============================================================================
def preprocess_frame(frame_bgr, size, device):
    resized = cv2.resize(frame_bgr, (size + 32, size + 32), interpolation=cv2.INTER_AREA)
    y0 = (resized.shape[0] - size) // 2
    x0 = (resized.shape[1] - size) // 2
    crop = resized[y0:y0 + size, x0:x0 + size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).to(device)


def preprocess_patch(patch_bgr, size, device):
    resized = cv2.resize(patch_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).to(device)


@torch.no_grad()
def detect_defect_fast(model, tensor):
    return torch.sigmoid(model(tensor)).item()


@torch.no_grad()
def build_defect_heatmap(model, frame_bgr, image_size, device, grid_size=3):
    h, w = frame_bgr.shape[:2]
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    ph, pw = h // grid_size, w // grid_size
    for gy in range(grid_size):
        for gx in range(grid_size):
            patch = frame_bgr[gy*ph:min((gy+1)*ph, h), gx*pw:min((gx+1)*pw, w)]
            if patch.shape[0] < 32 or patch.shape[1] < 32:
                continue
            tensor = preprocess_patch(patch, image_size, device)
            heatmap[gy, gx] = torch.sigmoid(model(tensor)).item()
    return heatmap


def draw_defect_circles(frame, heatmap, threshold, grid_size=3):
    h, w = frame.shape[:2]
    ph, pw = h // grid_size, w // grid_size
    overlay = frame.copy()
    for gy in range(grid_size):
        for gx in range(grid_size):
            if heatmap[gy, gx] >= threshold:
                cx, cy = gx * pw + pw // 2, gy * ph + ph // 2
                radius = min(pw, ph) // 2 - 5
                prob = heatmap[gy, gx]
                intensity = min(255, int(prob * 300))
                circ = overlay.copy()
                cv2.circle(circ, (cx, cy), radius, (0, 0, intensity), -1)
                cv2.addWeighted(circ, 0.3, overlay, 0.7, 0, overlay)
                cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), 2)
                cv2.putText(overlay, f"{prob:.0%}", (cx - 20, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


# ============================================================================
# PIPELINE STATE (shared across threads)
# ============================================================================
class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.class_idx = 0
        self.class_name = CLASS_ORDER[0]
        self.model = None
        self.meta = None
        self.paused = False
        self.run_heatmap = False
        self.frame_count = 0
        self.defect_count = 0
        self.good_count = 0
        self.current_fps = 0.0
        self.current_latency_ms = 0.0
        self.current_defect_prob = 0.0
        self.is_defect = False
        self.latest_frame_jpg = None
        self.stream_url = DEFAULT_STREAM_URL
        self.use_webcam = False

    def switch_class(self, class_idx):
        with self.lock:
            self.class_idx = class_idx
            self.class_name = CLASS_ORDER[class_idx]
            try:
                self.model, self.meta = load_defect_model(
                    MODEL_DIR, self.class_name, DEVICE)
                return True
            except FileNotFoundError:
                return False

    def get_status(self):
        with self.lock:
            return {
                "class_idx": self.class_idx,
                "class_name": self.class_name,
                "class_display": CLASS_DISPLAY_NAMES[self.class_name],
                "class_color": CLASS_COLORS_HEX[self.class_name],
                "paused": self.paused,
                "frame_count": self.frame_count,
                "defect_count": self.defect_count,
                "good_count": self.good_count,
                "fps": round(self.current_fps, 1),
                "latency_ms": round(self.current_latency_ms, 0),
                "defect_prob": round(self.current_defect_prob, 4),
                "is_defect": self.is_defect,
                "defect_rate": round(
                    self.defect_count / max(1, self.frame_count) * 100, 1),
                "threshold": float(self.meta.get("decision_threshold", 0.5)
                                   if self.meta else 0.5),
            }


state = PipelineState()


# ============================================================================
# CAPTURE & INFERENCE THREAD
# ============================================================================
def inference_loop():
    """Background thread: captures frames, runs inference, encodes JPEG."""
    # Load initial model
    print(f"Loading initial model: {state.class_name}")
    state.switch_class(0)

    # Open stream
    if state.use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(state.stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Cannot open video stream")
        return

    print("Video stream opened successfully")
    fps_buf = deque(maxlen=30)
    last_t = time.time()

    while True:
        if state.paused:
            time.sleep(0.1)
            continue

        # Flush buffer, grab latest
        for _ in range(2):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            if state.use_webcam:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(state.stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        t0 = time.time()

        with state.lock:
            model = state.model
            meta = state.meta

        if model is None or meta is None:
            continue

        img_size = int(meta.get("image_size", 288))
        threshold = float(meta.get("decision_threshold", 0.5))

        # Inference
        tensor = preprocess_frame(frame, img_size, DEVICE)
        defect_prob = detect_defect_fast(model, tensor)
        is_defect = defect_prob >= threshold

        # Heatmap on demand
        display = frame.copy()
        if state.run_heatmap and is_defect:
            heatmap = build_defect_heatmap(model, frame, img_size, DEVICE, 3)
            display = draw_defect_circles(display, heatmap, threshold, 3)
            state.run_heatmap = False
        elif state.run_heatmap:
            state.run_heatmap = False

        # Minimal annotation on the video frame itself
        status_text = "DEFECT" if is_defect else "GOOD"
        color = (0, 0, 255) if is_defect else (0, 220, 0)
        cv2.putText(display, f"{status_text} ({defect_prob:.1%})",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Timing
        t1 = time.time()
        latency = (t1 - t0) * 1000
        dt = t1 - last_t
        last_t = t1
        fps_buf.append(1.0 / max(dt, 0.001))

        # Update state
        with state.lock:
            state.frame_count += 1
            if is_defect:
                state.defect_count += 1
            else:
                state.good_count += 1
            state.current_fps = sum(fps_buf) / len(fps_buf)
            state.current_latency_ms = latency
            state.current_defect_prob = defect_prob
            state.is_defect = is_defect

        # Encode JPEG
        _, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        state.latest_frame_jpg = jpg.tobytes()


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="Textile Defect Detection")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassSelectRequest(BaseModel):
    class_idx: int


@app.on_event("startup")
async def startup():
    # Parse CLI-like env vars
    state.stream_url = os.environ.get("STREAM_URL", DEFAULT_STREAM_URL)
    state.use_webcam = os.environ.get("USE_WEBCAM", "0") == "1"

    # Create save dirs
    for f in ["good", "defect"]:
        os.makedirs(SAVE_BASE / f / "original", exist_ok=True)
        os.makedirs(SAVE_BASE / f / "processed", exist_ok=True)

    # Start inference thread
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()


@app.get("/api/status")
async def get_status():
    return JSONResponse(state.get_status())


@app.get("/api/classes")
async def get_classes():
    classes = []
    for i, name in enumerate(CLASS_ORDER):
        classes.append({
            "idx": i,
            "name": name,
            "display": CLASS_DISPLAY_NAMES[name],
            "description": CLASS_DESCRIPTIONS[name],
            "color": CLASS_COLORS_HEX[name],
        })
    return JSONResponse(classes)


@app.post("/api/select-class")
async def select_class(req: ClassSelectRequest):
    if not (0 <= req.class_idx < len(CLASS_ORDER)):
        return JSONResponse({"error": "Invalid class index"}, status_code=400)
    ok = state.switch_class(req.class_idx)
    if not ok:
        return JSONResponse({"error": "Model not found for this class"}, status_code=404)
    return JSONResponse({"ok": True, "class": CLASS_ORDER[req.class_idx]})


@app.post("/api/pause")
async def toggle_pause():
    state.paused = not state.paused
    return JSONResponse({"paused": state.paused})


@app.post("/api/heatmap")
async def trigger_heatmap():
    state.run_heatmap = True
    return JSONResponse({"ok": True})


@app.post("/api/reset-stats")
async def reset_stats():
    with state.lock:
        state.frame_count = 0
        state.defect_count = 0
        state.good_count = 0
    return JSONResponse({"ok": True})


@app.get("/api/video-feed")
async def video_feed():
    def generate():
        while True:
            if state.latest_frame_jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + state.latest_frame_jpg + b"\r\n")
            time.sleep(0.03)  # ~30fps cap

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    parser.add_argument("--use-webcam", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["STREAM_URL"] = args.stream_url
    if args.use_webcam:
        os.environ["USE_WEBCAM"] = "1"

    print(f"\n{'='*60}")
    print(f"  TEXTILE DEFECT DETECTION — WEB SERVER")
    print(f"  Stream: {'webcam' if args.use_webcam else args.stream_url}")
    print(f"  Device: {DEVICE}")
    print(f"  UI:     http://localhost:{args.port}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
