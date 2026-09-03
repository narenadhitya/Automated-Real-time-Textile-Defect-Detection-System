"""
train_base_classifier.py
========================
Trains a 4-class base classifier that identifies which textile pattern
an input image belongs to:
    0 → class_1_fine_texture
    1 → class_2_stochastic_texture
    2 → class_3_periodic_texture
    3 → class_4_printed_nonperiodic

This model is the first stage of the pipeline: it classifies the fabric type,
then routes to the corresponding per-class defect detection model.

Uses the same training strategy as the defect models:
  • EfficientNetV2-S backbone → 4-class softmax head
  • Phase 1: warmup (frozen backbone) → Phase 2: full fine-tune → Phase 3: SWA
  • CrossEntropy loss with class weighting
  • Cosine annealing + early stopping

Saves to: textile_models/base_classifier.pth + base_classifier_meta.json

Usage:
    python train_base_classifier.py
"""

import argparse
import json
import random
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
IMAGE_SIZE = 288
NUM_CLASSES = 4

CLASS_ORDER = [
    "class_1_fine_texture",
    "class_2_stochastic_texture",
    "class_3_periodic_texture",
    "class_4_printed_nonperiodic",
]

CLASS_DISPLAY_NAMES = {
    "class_1_fine_texture": "Fine Texture",
    "class_2_stochastic_texture": "Stochastic Texture",
    "class_3_periodic_texture": "Periodic Texture",
    "class_4_printed_nonperiodic": "Printed Non-Periodic",
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
# DATASET — 4-class classification
# ============================================================================
class FabricClassDataset(Dataset):
    """Each sample is (image_path, class_index) where class_index ∈ {0,1,2,3}."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ============================================================================
# DATA COLLECTION
# ============================================================================
def collect_all_samples(split_root: Path, split_name: str) -> List[Tuple[str, int]]:
    """Collect samples from all 4 classes, labeling by class index (0–3).
    Uses BOTH defect and no_defect images — the base model only cares about
    the textile pattern, not defect status."""
    out: List[Tuple[str, int]] = []
    for class_idx, class_name in enumerate(CLASS_ORDER):
        class_dir = split_root / class_name / split_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} does not exist, skipping")
            continue
        count = 0
        for subfolder in ["no_defect", "defect"]:
            folder = class_dir / subfolder
            if not folder.exists():
                continue
            for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
                for p in folder.rglob(ext):
                    out.append((str(p), class_idx))
                    count += 1
        print(f"  {class_name} ({split_name}): {count} images → label={class_idx}")
    return out


def split_train_val(samples: List[Tuple[str, int]], val_ratio: float, seed: int) -> Tuple[List, List]:
    """Stratified train/val split."""
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)

    rng = random.Random(seed)
    train_split: List[Tuple[str, int]] = []
    val_split: List[Tuple[str, int]] = []

    for label in sorted(by_label.keys()):
        items = by_label[label]
        rng.shuffle(items)
        n = len(items)
        n_val = max(2, int(n * val_ratio))
        if n_val >= n:
            n_val = n - 1
        val_split.extend(items[:n_val])
        train_split.extend(items[n_val:])

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


def build_sampler(samples: List[Tuple[str, int]], num_classes: int) -> WeightedRandomSampler:
    labels = np.array([y for _, y in samples], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = np.array([1.0 / max(1, c) for c in class_counts], dtype=np.float32)
    weights = np.array([class_weights[y] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)


# ============================================================================
# MODEL
# ============================================================================
def build_base_classifier(num_classes: int = 4) -> Tuple[nn.Module, list]:
    """EfficientNetV2-S with a 4-class softmax head."""
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.40, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.20),
        nn.Linear(256, num_classes),
    )
    backbone_modules = [model.features]
    return model, backbone_modules


# ============================================================================
# AUGMENTATION (same style as defect training)
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
# METRICS
# ============================================================================
def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    total = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = correct / total if total else 0.0

    # Per-class accuracy
    per_class = {}
    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_correct = int((y_pred[mask] == cls_idx).sum())
            per_class[cls_name] = cls_correct / int(mask.sum())
        else:
            per_class[cls_name] = 0.0

    return {"accuracy": acc, "correct": correct, "total": total, "per_class": per_class}


# ============================================================================
# TRAIN / EVAL
# ============================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_pred = []
    all_true = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(preds)
        all_true.append(y.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train base 4-class textile classifier")
    parser.add_argument("--split-root", type=str, default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--swa-epochs", type=int, default=8)
    args = parser.parse_args()

    split_root = Path(args.split_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print(f"{'='*60}")
    print(f"  BASE CLASSIFIER TRAINING")
    print(f"  Output: {out_dir}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ── Collect data ──
    print("\nCollecting training data...")
    train_full = collect_all_samples(split_root, "train")
    test_samples = collect_all_samples(split_root, "test")

    if len(train_full) < 50:
        raise RuntimeError(f"Insufficient training samples: {len(train_full)}")

    train_samples, val_samples = split_train_val(train_full, val_ratio=args.val_ratio, seed=SEED)
    print(f"\nSplit: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    # ── Transforms & loaders ──
    train_tf = get_train_transforms(IMAGE_SIZE)
    eval_tf = get_eval_transforms(IMAGE_SIZE)

    train_ds = FabricClassDataset(train_samples, transform=train_tf)
    val_ds = FabricClassDataset(val_samples, transform=eval_tf)
    test_ds = FabricClassDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=build_sampler(train_samples, NUM_CLASSES),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── Build model ──
    model, backbone_modules = build_base_classifier(NUM_CLASSES)
    model = model.to(device)

    # ── Class weights for CrossEntropy ──
    labels_arr = np.array([y for _, y in train_samples], dtype=np.int64)
    class_counts = np.bincount(labels_arr, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    weight_tensor = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # ── Phase 1: Warmup — freeze backbone ──
    print("\n── Phase 1: Warmup (frozen backbone) ──")
    for bm in backbone_modules:
        for p in bm.parameters():
            p.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=8e-4, weight_decay=1e-4)

    warmup_epochs = max(3, int(args.epochs * 0.15))
    for epoch in range(1, warmup_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"  WARMUP {epoch:02d}/{warmup_epochs} | loss={loss:.5f}")

    # ── Phase 2: Full fine-tuning ──
    print("\n── Phase 2: Full fine-tuning ──")
    for bm in backbone_modules:
        for p in bm.parameters():
            p.requires_grad = True

    all_backbone_params = []
    for bm in backbone_modules:
        all_backbone_params.extend(bm.parameters())
    head_params_full = [p for p in model.parameters() if p not in set(all_backbone_params)]

    optimizer = optim.AdamW([
        {"params": all_backbone_params, "lr": 8e-5},
        {"params": head_params_full, "lr": 8e-4},
    ], weight_decay=1e-4)

    main_epochs = args.epochs - warmup_epochs - args.swa_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(5, main_epochs // 4), T_mult=2, eta_min=1e-7,
    )

    model_path = out_dir / "base_classifier.pth"
    meta_path = out_dir / "base_classifier_meta.json"

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(warmup_epochs + 1, warmup_epochs + main_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()

        val_true, val_pred = evaluate(model, val_loader, device)
        val_metrics = compute_accuracy(val_true, val_pred)
        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"  EPOCH {epoch:03d} | loss={loss:.5f} | "
            f"val_acc={val_metrics['accuracy']:.4f} lr={lr_current:.2e}"
        )
        for cls_name, cls_acc in val_metrics["per_class"].items():
            print(f"    {cls_name}: {cls_acc:.4f}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"  EARLY_STOP at epoch {epoch} (patience={args.patience})")
            break

    # ── Phase 3: SWA ──
    print(f"\n── Phase 3: SWA ({args.swa_epochs} epochs) ──")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_optimizer = optim.SGD(model.parameters(), lr=4e-5, momentum=0.9, weight_decay=1e-4)
    swa_scheduler = optim.lr_scheduler.CosineAnnealingLR(swa_optimizer, T_max=args.swa_epochs, eta_min=1e-7)

    for swa_ep in range(1, args.swa_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, swa_optimizer, device, scaler)
        swa_model.update_parameters(model)
        swa_scheduler.step()
        print(f"  SWA {swa_ep:02d}/{args.swa_epochs} | loss={loss:.5f}")

    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # ── Compare best checkpoint vs SWA ──
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_true, test_pred_ckpt = evaluate(model, test_loader, device)
    ckpt_metrics = compute_accuracy(test_true, test_pred_ckpt)

    test_true_swa, test_pred_swa = evaluate(swa_model, test_loader, device)
    swa_metrics = compute_accuracy(test_true_swa, test_pred_swa)

    print(f"\n  COMPARE | ckpt_acc={ckpt_metrics['accuracy']:.4f} swa_acc={swa_metrics['accuracy']:.4f}")

    if swa_metrics["accuracy"] > ckpt_metrics["accuracy"]:
        print(f"  WINNER = SWA")
        torch.save(swa_model.module.state_dict(), model_path)
        final_metrics = swa_metrics
    else:
        print(f"  WINNER = best checkpoint (epoch {best_epoch})")
        final_metrics = ckpt_metrics

    # ── Save metadata ──
    meta = {
        "model_type": "base_classifier",
        "model_path": str(model_path),
        "backbone": "efficientnet_v2_s",
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "class_order": CLASS_ORDER,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "best_epoch": int(best_epoch),
        "test_accuracy": float(final_metrics["accuracy"]),
        "per_class_accuracy": {k: float(v) for k, v in final_metrics["per_class"].items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"BASE CLASSIFIER RESULTS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {final_metrics['accuracy']:.4f}")
    for cls_name, cls_acc in final_metrics["per_class"].items():
        print(f"  {cls_name}: {cls_acc:.4f}")
    print(f"  Model saved: {model_path}")
    print(f"  Meta saved:  {meta_path}")


if __name__ == "__main__":
    main()
