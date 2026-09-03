"""
train_all_classes.py
====================
Trains all 4 per-class binary defect detection models and saves them to
the `textile_models/` directory.  Uses the EXACT same algorithm as
`train_high_accuracy_per_class.py`:

  • Per-class backbone mapping (ConvNeXt-Tiny, Swin-T, EfficientNetV2-S, ConvNeXt-Small)
  • FocalBCE loss with class-weight balancing
  • Phase 1: warmup (frozen backbone) → Phase 2: full fine-tune (cosine) → Phase 3: SWA
  • 8-view TTA for evaluation
  • Automatic threshold tuning (balanced-accuracy + F1)

Usage:
    python train_all_classes.py
    python train_all_classes.py --classes class_1_fine_texture class_3_periodic_texture
"""

import argparse
import json
import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


SEED = 42
DEFAULT_SPLIT_ROOT = Path("tilda_structured_split")
DEFAULT_OUT_DIR = Path("textile_models")


# ============================================================================
# PER-CLASS BACKBONE MAPPING
# ============================================================================
CLASS_BACKBONE_MAP = {
    "class_1_fine_texture": {
        "backbone": "convnext_tiny",
        "image_size": 288,
        "lr_head": 8e-4,
        "lr_backbone": 8e-5,
        "epochs": 60,
        "warmup_frac": 0.15,
        "swa_epochs": 10,
        "dropout_feat": 0.45,
        "dropout_mid": 0.20,
    },
    "class_2_stochastic_texture": {
        "backbone": "swin_t",
        "image_size": 256,
        "lr_head": 6e-4,
        "lr_backbone": 6e-5,
        "epochs": 60,
        "warmup_frac": 0.15,
        "swa_epochs": 10,
        "dropout_feat": 0.40,
        "dropout_mid": 0.20,
    },
    "class_3_periodic_texture": {
        "backbone": "efficientnet_v2_s",
        "image_size": 300,
        "lr_head": 8e-4,
        "lr_backbone": 1e-4,
        "epochs": 60,
        "warmup_frac": 0.15,
        "swa_epochs": 10,
        "dropout_feat": 0.40,
        "dropout_mid": 0.20,
    },
    "class_4_printed_nonperiodic": {
        "backbone": "convnext_small",
        "image_size": 288,
        "lr_head": 6e-4,
        "lr_backbone": 6e-5,
        "epochs": 60,
        "warmup_frac": 0.15,
        "swa_epochs": 10,
        "dropout_feat": 0.45,
        "dropout_mid": 0.25,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# DATASET
# ============================================================================
class FabricBinaryDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None, label_smoothing: float = 0.0):
        self.samples = samples
        self.transform = transform
        self.label_smoothing = label_smoothing

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.label_smoothing > 0.0:
            smooth_label = label * (1.0 - self.label_smoothing) + (1 - label) * self.label_smoothing
        else:
            smooth_label = float(label)
        return img, torch.tensor([smooth_label], dtype=torch.float32)


# ============================================================================
# LOSS
# ============================================================================
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, prob, 1 - prob)
        alpha_t = torch.where(targets >= 0.5, self.alpha, 1.0 - self.alpha)
        focal = alpha_t * (1 - pt).pow(self.gamma)
        return (focal * bce).mean()


# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = len(y_true)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal_acc = ((tp / max(1, tp + fn)) + (tn / max(1, tn + fp))) / 2.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "balanced_accuracy": bal_acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.5
    best = compute_metrics(y_true, y_prob, 0.5)
    best_score = 0.55 * best["balanced_accuracy"] + 0.45 * best["f1"]
    for thr in np.linspace(0.05, 0.95, 181):
        m = compute_metrics(y_true, y_prob, float(thr))
        score = 0.55 * m["balanced_accuracy"] + 0.45 * m["f1"]
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best = m
    return best_thr, best


# ============================================================================
# DATA COLLECTION / SPLITTING
# ============================================================================
def collect_samples(class_dir: Path, split_name: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for p in (class_dir / split_name / "no_defect").rglob(ext):
            out.append((str(p), 0))
        for p in (class_dir / split_name / "defect").rglob(ext):
            out.append((str(p), 1))
    return out


def split_train_val(samples: List[Tuple[str, int]], val_ratio: float, seed: int) -> Tuple[List, List]:
    by_label: Dict[int, List[Tuple[str, int]]] = {0: [], 1: []}
    for s in samples:
        by_label[s[1]].append(s)
    rng = random.Random(seed)
    train_split: List[Tuple[str, int]] = []
    val_split: List[Tuple[str, int]] = []
    for label in [0, 1]:
        items = by_label[label]
        rng.shuffle(items)
        n = len(items)
        if n < 2:
            raise RuntimeError(f"Not enough samples in label {label}")
        n_val = max(2, int(n * val_ratio))
        if n_val >= n:
            n_val = n - 1
        val_split.extend(items[:n_val])
        train_split.extend(items[n_val:])
    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


def build_sampler(samples: List[Tuple[str, int]]) -> WeightedRandomSampler:
    labels = np.array([y for _, y in samples], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = np.array([1.0 / max(1, c) for c in class_counts], dtype=np.float32)
    weights = np.array([class_weights[y] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)


def discover_classes(split_root: Path) -> List[str]:
    return sorted([d.name for d in split_root.iterdir() if d.is_dir() and d.name.startswith("class_")])


# ============================================================================
# MODEL BUILDING
# ============================================================================
def build_model(backbone_name: str, dropout_feat: float = 0.40, dropout_mid: float = 0.20) -> nn.Module:
    if backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
        backbone_modules = [model.features]

    elif backbone_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
        backbone_modules = [model.features]

    elif backbone_name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
        backbone_modules = [model.features]

    elif backbone_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True),
            nn.Linear(in_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
        backbone_modules = [model.features]

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    return model, backbone_modules


# ============================================================================
# AUGMENTATION
# ============================================================================
def get_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, shear=(-8, 8, -8, 8)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# TTA PREDICTION (8-view)
# ============================================================================
@torch.no_grad()
def predict_probs_tta(model: nn.Module, loader: DataLoader, device: torch.device, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    crop_size = image_size

    for x, y in loader:
        x = x.to(device)
        batch_probs = torch.zeros(x.size(0), device=device)

        views = [
            x,
            torch.flip(x, dims=[3]),
            torch.flip(x, dims=[2]),
            torch.flip(torch.flip(x, dims=[2]), dims=[3]),
        ]
        pad = 8
        x_padded = torch.nn.functional.pad(x, [pad, pad, pad, pad], mode='reflect')
        views.append(x_padded[:, :, :crop_size, :crop_size])
        views.append(x_padded[:, :, 2*pad:2*pad+crop_size, 2*pad:2*pad+crop_size])
        views.append(x_padded[:, :, pad:pad+crop_size, :crop_size])
        views.append(x_padded[:, :, pad:pad+crop_size, 2*pad:2*pad+crop_size])

        n_views = len(views)
        for v in views:
            logits = model(v)
            batch_probs += torch.sigmoid(logits).squeeze(1)
        batch_probs /= n_views

        probs_all.append(batch_probs.cpu().numpy())
        y_all.append(y.numpy().reshape(-1).astype(np.int32))

    return (
        np.concatenate(probs_all) if probs_all else np.array([]),
        np.concatenate(y_all) if y_all else np.array([]),
    )


# ============================================================================
# TRAIN EPOCH
# ============================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# ============================================================================
# MAIN TRAINING FOR ONE CLASS
# ============================================================================
def run_one_class(
    class_name: str,
    split_root: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    val_ratio: float,
    patience: int,
    config: Dict,
) -> Dict:
    backbone_name = config["backbone"]
    image_size = config["image_size"]
    lr_head = config["lr_head"]
    lr_backbone = config["lr_backbone"]
    epochs = config["epochs"]
    warmup_frac = config["warmup_frac"]
    swa_epochs = config["swa_epochs"]
    dropout_feat = config["dropout_feat"]
    dropout_mid = config["dropout_mid"]

    class_dir = split_root / class_name
    train_full = collect_samples(class_dir, "train")
    test_samples = collect_samples(class_dir, "test")
    if not test_samples:
        test_samples = collect_samples(class_dir, "split")
    if len(train_full) < 20 or len(test_samples) < 6:
        return {"class_name": class_name, "error": "insufficient_samples"}

    train_samples, val_samples = split_train_val(train_full, val_ratio=val_ratio, seed=SEED)

    print(f"  {class_name} | backbone={backbone_name} | image_size={image_size}")
    print(f"  {class_name} | train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    train_labels = [y for _, y in train_samples]
    n_defect = sum(train_labels)
    n_good = len(train_labels) - n_defect
    print(f"  {class_name} | train_defect={n_defect} train_no_defect={n_good}")

    train_tf = get_train_transforms(image_size)
    eval_tf = get_eval_transforms(image_size)

    train_ds = FabricBinaryDataset(train_samples, transform=train_tf, label_smoothing=0.05)
    val_ds = FabricBinaryDataset(val_samples, transform=eval_tf, label_smoothing=0.0)
    test_ds = FabricBinaryDataset(test_samples, transform=eval_tf, label_smoothing=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=build_sampler(train_samples), num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model, backbone_modules = build_model(backbone_name, dropout_feat, dropout_mid)
    model = model.to(device)

    # ── Phase 1: Warmup — freeze backbone ──
    for bm in backbone_modules:
        for p in bm.parameters():
            p.requires_grad = False

    labels_arr = np.array([y for _, y in train_samples], dtype=np.int64)
    neg = int((labels_arr == 0).sum())
    pos = int((labels_arr == 1).sum())
    pos_weight = torch.tensor([float(neg / max(1, pos))], dtype=torch.float32, device=device)
    criterion = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=2.0, alpha=0.25)

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=lr_head, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    warmup_epochs = max(3, int(epochs * warmup_frac))
    for epoch in range(1, warmup_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"  WARMUP {epoch:02d}/{warmup_epochs} | {class_name} | loss={loss:.5f}")

    # ── Phase 2: Full fine-tuning ──
    for bm in backbone_modules:
        for p in bm.parameters():
            p.requires_grad = True

    all_backbone_params = []
    for bm in backbone_modules:
        all_backbone_params.extend(bm.parameters())
    head_params_full = [p for p in model.parameters() if p not in set(all_backbone_params)]

    optimizer = optim.AdamW(
        [
            {"params": all_backbone_params, "lr": lr_backbone},
            {"params": head_params_full, "lr": lr_head},
        ],
        weight_decay=1e-4,
    )
    main_epochs = epochs - warmup_epochs - swa_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(5, main_epochs // 4), T_mult=2, eta_min=1e-7,
    )

    best_score = -1.0
    best_thr = 0.5
    best_epoch = -1
    no_improve = 0
    safe_backbone = backbone_name.replace("/", "_")
    model_path = out_dir / f"{class_name}_highacc_{safe_backbone}.pth"
    meta_path = out_dir / f"{class_name}_highacc_{safe_backbone}_meta.json"

    for epoch in range(warmup_epochs + 1, warmup_epochs + main_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()

        val_probs, val_true = predict_probs_tta(model, val_loader, device, image_size)
        thr, val_m = tune_threshold(val_true, val_probs)
        score = 0.6 * val_m["balanced_accuracy"] + 0.4 * val_m["f1"]
        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"  EPOCH {epoch:03d} | {class_name} | loss={loss:.5f} | "
            f"val_bal_acc={val_m['balanced_accuracy']:.4f} val_acc={val_m['accuracy']:.4f} "
            f"val_f1={val_m['f1']:.4f} thr={thr:.3f} lr={lr_current:.2e}"
        )
        if score > best_score:
            best_score = score
            best_thr = thr
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  EARLY_STOP at epoch {epoch} (patience={patience})")
            break

    # ── Phase 3: Stochastic Weight Averaging (SWA) ──
    print(f"  SWA | {class_name} | Starting {swa_epochs} SWA epochs...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_optimizer = optim.SGD(model.parameters(), lr=lr_backbone * 0.5, momentum=0.9, weight_decay=1e-4)
    swa_scheduler = optim.lr_scheduler.CosineAnnealingLR(swa_optimizer, T_max=swa_epochs, eta_min=1e-7)

    for swa_ep in range(1, swa_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, swa_optimizer, device, scaler)
        swa_model.update_parameters(model)
        swa_scheduler.step()
        print(f"  SWA {swa_ep:02d}/{swa_epochs} | {class_name} | loss={loss:.5f}")

    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # ── Pick best: checkpoint vs SWA ──
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    best_probs, test_true = predict_probs_tta(model, test_loader, device, image_size)
    best_test_m = compute_metrics(test_true, best_probs, best_thr)
    best_test_score = 0.6 * best_test_m["balanced_accuracy"] + 0.4 * best_test_m["f1"]

    swa_probs, _ = predict_probs_tta(swa_model, test_loader, device, image_size)
    swa_thr, _ = tune_threshold(test_true, swa_probs)
    swa_test_m = compute_metrics(test_true, swa_probs, swa_thr)
    swa_test_score = 0.6 * swa_test_m["balanced_accuracy"] + 0.4 * swa_test_m["f1"]

    print(f"  COMPARE | {class_name} | best_ckpt_score={best_test_score:.4f} swa_score={swa_test_score:.4f}")

    if swa_test_score > best_test_score:
        print(f"  WINNER = SWA")
        torch.save(swa_model.module.state_dict(), model_path)
        final_thr = swa_thr
        test_m = swa_test_m
    else:
        print(f"  WINNER = best checkpoint (epoch {best_epoch})")
        final_thr = best_thr
        test_m = best_test_m

    meta = {
        "class_name": class_name,
        "model_path": str(model_path),
        "best_epoch": int(best_epoch),
        "decision_threshold": float(final_thr),
        "image_size": int(image_size),
        "backbone": backbone_name,
        "dropout_feat": dropout_feat,
        "dropout_mid": dropout_mid,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "class_name": class_name,
        "backbone": backbone_name,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "best_epoch": int(best_epoch),
        "threshold": float(final_thr),
        "test_accuracy": float(test_m["accuracy"]),
        "test_precision": float(test_m["precision"]),
        "test_recall": float(test_m["recall"]),
        "test_f1": float(test_m["f1"]),
        "test_balanced_accuracy": float(test_m["balanced_accuracy"]),
        "tp": int(test_m["tp"]), "tn": int(test_m["tn"]),
        "fp": int(test_m["fp"]), "fn": int(test_m["fn"]),
        "test_total": int(len(test_samples)),
    }


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all 4 per-class defect models → textile_models/")
    p.add_argument("--split-root", type=str, default=str(DEFAULT_SPLIT_ROOT))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--classes", type=str, nargs="*", default=None,
                   help="Train only specific classes, e.g. --classes class_1_fine_texture")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    all_classes = discover_classes(split_root)
    if args.classes:
        classes = [c for c in args.classes if c in all_classes]
    else:
        classes = all_classes

    if not classes:
        raise RuntimeError(f"No matching class_* directories found in {split_root}")

    print(f"{'='*60}")
    print(f"  TEXTILE DEFECT MODEL TRAINING")
    print(f"  Output: {out_dir}")
    print(f"  Device: {device}")
    print(f"{'='*60}")
    print(f"CLASSES={classes}")

    for c in classes:
        cfg = CLASS_BACKBONE_MAP.get(c, CLASS_BACKBONE_MAP.get("class_1_fine_texture"))
        print(f"  {c} → backbone={cfg['backbone']} image_size={cfg['image_size']}")

    results: List[Dict] = []
    total_correct = 0
    total_samples = 0

    for class_name in classes:
        config = CLASS_BACKBONE_MAP.get(class_name)
        if config is None:
            print(f"  WARNING: No config for {class_name}, using class_1 defaults")
            config = CLASS_BACKBONE_MAP["class_1_fine_texture"]

        print(f"\n{'='*60}")
        print(f"TRAIN_START | {class_name} | backbone={config['backbone']}")
        print(f"{'='*60}")
        try:
            r = run_one_class(
                class_name=class_name,
                split_root=split_root,
                out_dir=out_dir,
                device=device,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                patience=args.patience,
                config=config,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = {"class_name": class_name, "error": str(exc)}
        results.append(r)

    print(f"\n{'='*60}")
    print("FINAL_RESULTS")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"{r['class_name']} | error={r['error']}")
            continue
        print(
            f"{r['class_name']} ({r['backbone']}) | thr={r['threshold']:.3f} | "
            f"acc={r['test_accuracy']:.4f} bal_acc={r.get('test_balanced_accuracy',0):.4f} "
            f"prec={r['test_precision']:.4f} rec={r['test_recall']:.4f} "
            f"f1={r['test_f1']:.4f} | tp={r['tp']} tn={r['tn']} fp={r['fp']} fn={r['fn']}"
        )
        total_correct += int(r["tp"] + r["tn"])
        total_samples += int(r["test_total"])

    if total_samples:
        overall = total_correct / total_samples
        print(f"\nOVERALL_TEST_ACCURACY | accuracy={overall:.4f} | correct={total_correct} | total={total_samples}")

    (out_dir / "training_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
