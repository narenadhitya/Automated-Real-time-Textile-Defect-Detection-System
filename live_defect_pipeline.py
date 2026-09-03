"""
live_defect_pipeline.py
=======================
Live-stream textile defect detection with enhanced GUI.

FAST PIPELINE (1-2 sec/frame):
  1. User selects fabric class at startup (no per-frame classifier)
  2. Single-pass defect inference (no TTA for speed)
  3. Heatmap localization only on-demand (press H)
  4. Buffer-flushing to stay real-time

Usage:
    python live_defect_pipeline.py --class 1
    python live_defect_pipeline.py --class 2 --stream-url http://192.168.1.5:8080/video
    python live_defect_pipeline.py --class 3 --use-webcam
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_STREAM_URL = "http://10.53.181.62:8080/video"
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

# Color palette for each class (BGR)
CLASS_COLORS = {
    "class_1_fine_texture": (255, 180, 50),
    "class_2_stochastic_texture": (50, 220, 220),
    "class_3_periodic_texture": (50, 200, 50),
    "class_4_printed_nonperiodic": (200, 100, 255),
}

BACKBONE_NAMES = ["convnext_tiny", "convnext_small", "swin_t",
                  "efficientnet_v2_s", "efficientnet_b3"]

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ============================================================================
# LOGGING
# ============================================================================
def log_message(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


# ============================================================================
# MODEL BUILDING (matches training architectures exactly)
# ============================================================================
def build_defect_model(backbone: str, dropout_feat: float = 0.40, dropout_mid: float = 0.20) -> nn.Module:
    """Build a binary defect detection model (matches training architecture)."""
    if backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "convnext_small":
        model = models.convnext_small(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "swin_t":
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True),
            nn.Linear(in_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True),
            nn.Linear(in_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ============================================================================
# LOAD SINGLE CLASS MODEL (fast — only loads what's needed)
# ============================================================================
def load_defect_model(model_dir: Path, class_name: str, device: str):
    """Load one defect model + its metadata. Returns (model, meta) or raises."""
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
        raise FileNotFoundError(f"No defect model found for {class_name} in {model_dir}")

    # Load metadata
    meta = {"backbone": "convnext_tiny", "image_size": 288,
            "decision_threshold": 0.5, "dropout_feat": 0.45, "dropout_mid": 0.20}
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta.update(json.load(f))

    # Build and load model
    model = build_defect_model(
        meta["backbone"],
        meta.get("dropout_feat", 0.45),
        meta.get("dropout_mid", 0.20),
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    print(f"[OK] Loaded {class_name}: backbone={meta['backbone']}, "
          f"threshold={meta['decision_threshold']:.3f}, size={meta['image_size']}")
    return model, meta


# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess_frame(frame_bgr: np.ndarray, size: int, device: str) -> torch.Tensor:
    """Preprocess a BGR frame for model inference."""
    resized = cv2.resize(frame_bgr, (size + 32, size + 32), interpolation=cv2.INTER_AREA)
    y0 = (resized.shape[0] - size) // 2
    x0 = (resized.shape[1] - size) // 2
    crop = resized[y0:y0 + size, x0:x0 + size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    return tensor


def preprocess_patch(patch_bgr: np.ndarray, size: int, device: str) -> torch.Tensor:
    """Preprocess a small BGR patch for defect localization."""
    resized = cv2.resize(patch_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    return tensor


# ============================================================================
# INFERENCE — SINGLE PASS (fast, no TTA)
# ============================================================================
@torch.no_grad()
def detect_defect_fast(model: nn.Module, tensor: torch.Tensor) -> float:
    """Single-pass inference. Returns defect probability."""
    return torch.sigmoid(model(tensor)).item()


@torch.no_grad()
def build_defect_heatmap(model: nn.Module, frame_bgr: np.ndarray,
                         image_size: int, device: str,
                         grid_size: int = 3) -> np.ndarray:
    """
    Sliding-window defect heatmap. Divides frame into grid_size×grid_size
    patches. Returns a probability heatmap.
    """
    h, w = frame_bgr.shape[:2]
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    patch_h = h // grid_size
    patch_w = w // grid_size

    for gy in range(grid_size):
        for gx in range(grid_size):
            y1 = gy * patch_h
            x1 = gx * patch_w
            y2 = min(y1 + patch_h, h)
            x2 = min(x1 + patch_w, w)
            patch = frame_bgr[y1:y2, x1:x2]
            if patch.shape[0] < 32 or patch.shape[1] < 32:
                continue
            tensor = preprocess_patch(patch, image_size, device)
            prob = torch.sigmoid(model(tensor)).item()
            heatmap[gy, gx] = prob

    return heatmap


def draw_defect_regions(frame: np.ndarray, heatmap: np.ndarray,
                        threshold: float, grid_size: int = 3) -> np.ndarray:
    """Draw circles around high-probability defect regions."""
    h, w = frame.shape[:2]
    patch_h = h // grid_size
    patch_w = w // grid_size
    overlay = frame.copy()

    for gy in range(grid_size):
        for gx in range(grid_size):
            if heatmap[gy, gx] >= threshold:
                cx = gx * patch_w + patch_w // 2
                cy = gy * patch_h + patch_h // 2
                radius = min(patch_w, patch_h) // 2 - 5
                prob = heatmap[gy, gx]
                intensity = min(255, int(prob * 300))

                # Semi-transparent filled circle
                circle_overlay = overlay.copy()
                cv2.circle(circle_overlay, (cx, cy), radius, (0, 0, intensity), -1)
                cv2.addWeighted(circle_overlay, 0.25, overlay, 0.75, 0, overlay)

                # Circle border
                cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), 2)

                # Probability text
                cv2.putText(overlay, f"{prob:.0%}", (cx - 20, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

    return overlay


# ============================================================================
# ENHANCED GUI DRAWING
# ============================================================================
def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=0.7):
    """Draw a rounded rectangle with transparency."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_confidence_bar(img, x, y, w, h, value, max_val=1.0,
                        color_low=(0, 0, 200), color_high=(0, 200, 0)):
    """Draw a horizontal confidence/progress bar."""
    ratio = min(value / max_val, 1.0)
    filled_w = int(w * ratio)
    r = int(color_low[0] + (color_high[0] - color_low[0]) * ratio)
    g = int(color_low[1] + (color_high[1] - color_low[1]) * ratio)
    b = int(color_low[2] + (color_high[2] - color_low[2]) * ratio)
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    if filled_w > 0:
        cv2.rectangle(img, (x, y), (x + filled_w, y + h), (b, g, r), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (150, 150, 150), 1)


def draw_hud_panel(frame: np.ndarray, stats: dict) -> np.ndarray:
    """Draw a comprehensive HUD overlay on the frame."""
    h, w = frame.shape[:2]
    output = frame.copy()

    # ──────── Top Status Bar ────────
    bar_h = 50
    draw_rounded_rect(output, (0, 0), (w, bar_h), (30, 30, 30), radius=0, alpha=0.8)

    cv2.putText(output, "TEXTILE DEFECT DETECTION",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 255), 2, cv2.LINE_AA)

    # FPS + latency on right
    fps_text = f"FPS: {stats.get('fps', 0):.1f}"
    latency_text = f"Latency: {stats.get('latency_ms', 0):.0f}ms"
    cv2.putText(output, fps_text, (w - 260, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(output, latency_text, (w - 260, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)

    # ──────── Left Info Panel ────────
    panel_w = 280
    panel_h = 230
    panel_y = 60
    draw_rounded_rect(output, (8, panel_y), (panel_w, panel_y + panel_h),
                      (20, 20, 20), radius=10, alpha=0.75)

    y_off = panel_y + 25
    line_h = 28

    # Fabric class (user-selected, fixed)
    class_display = stats.get("class_display", "Unknown")
    class_color = stats.get("class_color", (200, 200, 200))
    cv2.putText(output, "FABRIC CLASS (SELECTED)", (18, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
    y_off += 22
    cv2.putText(output, class_display, (18, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color, 2, cv2.LINE_AA)
    y_off += line_h + 8

    # Defect status
    is_defect = stats.get("is_defect", False)
    defect_prob = stats.get("defect_prob", 0)
    status_text = "DEFECT DETECTED!" if is_defect else "NO DEFECT"
    status_color = (0, 0, 255) if is_defect else (0, 220, 0)
    cv2.putText(output, "STATUS", (18, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
    y_off += 22
    cv2.putText(output, status_text, (18, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
    y_off += line_h + 2

    # Defect probability bar
    cv2.putText(output, "DEFECT PROBABILITY", (18, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
    y_off += 18
    draw_confidence_bar(output, 18, y_off, 245, 16, defect_prob, 1.0,
                        color_low=(50, 200, 50), color_high=(50, 50, 200))
    cv2.putText(output, f"{defect_prob:.1%}", (200, y_off + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    y_off += 30

    # Frame count
    cv2.putText(output, f"Frame: {stats.get('frame_count', 0)}",
                (18, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)

    # ──────── Right Statistics Panel ────────
    stat_panel_w = 220
    stat_panel_x = w - stat_panel_w - 8
    stat_panel_y = 60
    stat_panel_h = 160
    draw_rounded_rect(output, (stat_panel_x, stat_panel_y),
                      (w - 8, stat_panel_y + stat_panel_h),
                      (20, 20, 20), radius=10, alpha=0.75)

    y_off = stat_panel_y + 25
    cv2.putText(output, "SESSION STATS", (stat_panel_x + 12, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 200, 100), 1, cv2.LINE_AA)
    y_off += line_h

    total_frames = stats.get("total_inspected", 0)
    defect_count = stats.get("defect_count", 0)
    good_count = stats.get("good_count", 0)
    defect_rate = defect_count / max(1, total_frames) * 100

    lines = [
        (f"Inspected: {total_frames}", (200, 200, 200)),
        (f"Defects: {defect_count}", (100, 100, 255)),
        (f"Good: {good_count}", (100, 255, 100)),
        (f"Defect Rate: {defect_rate:.1f}%", (220, 180, 100)),
    ]
    for text, color in lines:
        cv2.putText(output, text, (stat_panel_x + 15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        y_off += line_h

    # ──────── Bottom Status Bar ────────
    bottom_bar_y = h - 35
    draw_rounded_rect(output, (0, bottom_bar_y), (w, h), (30, 30, 30), radius=0, alpha=0.8)

    indicator_color = (0, 0, 200) if is_defect else (0, 180, 0)
    cv2.rectangle(output, (0, bottom_bar_y), (6, h), indicator_color, -1)

    cv2.putText(output, "Q:Quit  S:Screenshot  P:Pause  H:Heatmap  1-4:Switch Class",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (180, 180, 180), 1, cv2.LINE_AA)

    ts = time.strftime("%H:%M:%S")
    cv2.putText(output, ts, (w - 80, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    return output


# ============================================================================
# CLASS SELECTION MENU
# ============================================================================
def select_class_interactive() -> int:
    """Print menu and let user pick a class. Returns class index (0-3)."""
    print("\n" + "=" * 60)
    print("  SELECT FABRIC CLASS")
    print("=" * 60)
    for i, cls_name in enumerate(CLASS_ORDER):
        display = CLASS_DISPLAY_NAMES[cls_name]
        print(f"  [{i+1}] {display}  ({cls_name})")
    print("=" * 60)

    while True:
        try:
            choice = int(input("\n  Enter class number (1-4): ").strip())
            if 1 <= choice <= 4:
                return choice - 1
        except (ValueError, EOFError):
            pass
        print("  Invalid choice. Enter 1, 2, 3, or 4.")


# ============================================================================
# FLUSH STREAM BUFFER — grabs the most recent frame
# ============================================================================
def grab_latest_frame(cap: cv2.VideoCapture) -> tuple:
    """Flush old buffered frames and return only the latest one."""
    # Grab (don't decode) several frames to clear the buffer
    for _ in range(3):
        cap.grab()
    # Now read the latest
    ret, frame = cap.read()
    return ret, frame


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Live Textile Defect Detection Pipeline")
    parser.add_argument("--stream-url", type=str, default=DEFAULT_STREAM_URL,
                        help="Mobile camera stream URL")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--use-webcam", action="store_true",
                        help="Use local webcam instead of mobile stream")
    parser.add_argument("--class", type=int, default=0, dest="fabric_class",
                        help="Fabric class to use (1-4). 0 = interactive menu.")
    parser.add_argument("--save-interval", type=int, default=30,
                        help="Save frame every N frames")
    parser.add_argument("--heatmap-grid", type=int, default=3,
                        help="Grid size for defect heatmap (NxN, default 3)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # ── Select class ──
    if 1 <= args.fabric_class <= 4:
        class_idx = args.fabric_class - 1
    else:
        class_idx = select_class_interactive()

    class_name = CLASS_ORDER[class_idx]
    class_display = CLASS_DISPLAY_NAMES[class_name]
    class_color = CLASS_COLORS[class_name]
    print(f"\n  Selected: [{class_idx+1}] {class_display}")

    # ── Create save directories ──
    for cls_folder in ["good", "defect"]:
        os.makedirs(SAVE_BASE / cls_folder / "original", exist_ok=True)
        os.makedirs(SAVE_BASE / cls_folder / "processed", exist_ok=True)

    # ── Load ONLY the selected class model (fast startup) ──
    print(f"\n  Loading model for {class_name}...")
    defect_model, defect_meta = load_defect_model(model_dir, class_name, DEVICE)
    img_size = int(defect_meta.get("image_size", 288))
    threshold = float(defect_meta.get("decision_threshold", 0.5))

    # ── Open stream ──
    if args.use_webcam:
        print("\nOpening local webcam...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"\nConnecting to mobile stream: {args.stream_url}")
        cap = cv2.VideoCapture(args.stream_url)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Cannot connect to video source")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  LIVE STREAM STARTED  —  Press Q to quit")
    print("  Controls: Q=Quit  S=Screenshot  P=Pause  H=Heatmap  1-4=Switch Class")
    print("=" * 60 + "\n")

    # ── State tracking ──
    frame_count = 0
    defect_count = 0
    good_count = 0
    fps_buffer = deque(maxlen=30)
    paused = False
    run_heatmap_next = False  # heatmap is on-demand only
    last_time = time.time()
    last_heatmap = None
    last_heatmap_threshold = threshold
    display_frame = None

    # Window setup
    window_name = "Textile Defect Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        if not paused:
            # ── Grab latest frame (flush buffer for low latency) ──
            t_start = time.time()
            ret, frame = grab_latest_frame(cap)
            if not ret:
                print("Stream error — attempting reconnect...")
                time.sleep(1)
                if args.use_webcam:
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(args.stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            # ── Single-pass defect inference (FAST) ──
            defect_tensor = preprocess_frame(frame, img_size, DEVICE)
            defect_prob = detect_defect_fast(defect_model, defect_tensor)
            is_defect = defect_prob >= threshold

            # ── Timing ──
            t_infer = time.time()
            latency_ms = (t_infer - t_start) * 1000
            dt = t_infer - last_time
            last_time = t_infer
            fps_buffer.append(1.0 / max(dt, 0.001))
            current_fps = sum(fps_buffer) / len(fps_buffer)

            # ── Heatmap (on-demand only — press H) ──
            display_frame = frame.copy()

            if run_heatmap_next and is_defect:
                print("  Computing heatmap...")
                heatmap = build_defect_heatmap(
                    defect_model, frame, img_size, DEVICE,
                    grid_size=args.heatmap_grid
                )
                display_frame = draw_defect_regions(
                    display_frame, heatmap, threshold,
                    grid_size=args.heatmap_grid
                )
                last_heatmap = heatmap
                last_heatmap_threshold = threshold
                run_heatmap_next = False
            elif run_heatmap_next and not is_defect:
                print("  No defect detected — skipping heatmap")
                run_heatmap_next = False

            # ── Update stats ──
            frame_count += 1
            if is_defect:
                defect_count += 1
            else:
                good_count += 1

            # ── Draw HUD ──
            stats = {
                "fps": current_fps,
                "latency_ms": latency_ms,
                "class_display": class_display,
                "class_color": class_color,
                "is_defect": is_defect,
                "defect_prob": defect_prob,
                "frame_count": frame_count,
                "total_inspected": frame_count,
                "defect_count": defect_count,
                "good_count": good_count,
            }
            display_frame = draw_hud_panel(display_frame, stats)

            # ── Show frame ──
            cv2.imshow(window_name, display_frame)

            # ── Periodic save & log ──
            if frame_count % args.save_interval == 0:
                cls_folder = "defect" if is_defect else "good"
                timestamp = int(time.time() * 1000)
                orig_path = SAVE_BASE / cls_folder / "original" / f"{cls_folder}_{timestamp}.jpg"
                proc_path = SAVE_BASE / cls_folder / "processed" / f"{cls_folder}_{timestamp}.jpg"
                cv2.imwrite(str(orig_path), frame)
                cv2.imwrite(str(proc_path), display_frame)

                log_message(f"Frame={frame_count} | Class={class_display} | "
                            f"Defect={is_defect} ({defect_prob:.4f}) | "
                            f"Latency={latency_ms:.0f}ms | D/G={defect_count}/{good_count}")

        # ── Keyboard controls ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and display_frame is not None:
            ts = int(time.time() * 1000)
            screenshot_path = SAVE_BASE / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(screenshot_path), display_frame)
            print(f"  Screenshot saved: {screenshot_path}")
        elif key == ord("p"):
            paused = not paused
            print(f"  {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord("h"):
            run_heatmap_next = True
            print("  Heatmap will run on next frame...")
        elif key in [ord("1"), ord("2"), ord("3"), ord("4")]:
            new_idx = key - ord("1")
            if new_idx != class_idx:
                class_idx = new_idx
                class_name = CLASS_ORDER[class_idx]
                class_display = CLASS_DISPLAY_NAMES[class_name]
                class_color = CLASS_COLORS[class_name]
                print(f"\n  Switching to [{class_idx+1}] {class_display}...")
                try:
                    defect_model, defect_meta = load_defect_model(model_dir, class_name, DEVICE)
                    img_size = int(defect_meta.get("image_size", 288))
                    threshold = float(defect_meta.get("decision_threshold", 0.5))
                except FileNotFoundError as e:
                    print(f"  ERROR: {e}")

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Class: {class_display}")
    print(f"  Total Frames: {frame_count}")
    print(f"  Defects: {defect_count} ({defect_count/max(1,frame_count)*100:.1f}%)")
    print(f"  Good: {good_count} ({good_count/max(1,frame_count)*100:.1f}%)")


if __name__ == "__main__":
    main()
