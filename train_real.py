#!/usr/bin/env python3
"""Reproducible real-image training for the four-class household-waste task.
This file is part of the public GitHub Actions training workflow.
"""
import argparse, copy, hashlib, json, os, random, shutil, time, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = False
SOURCE_DATASET = "huaweilin/waste-classification"
SOURCE_SPLIT = "cleaned"
CLASSES = ["hazardous", "kitchen", "other", "recyclable"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
SOURCE_TO_CLASS = {
    "Hazardous": "hazardous",
    "Organic": "kitchen",
    "Non-Recyclable": "other",
    "Recyclable": "recyclable",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SKAttention(nn.Module):
    """Lightweight two-branch selective-kernel attention block for 1280-channel features."""
    def __init__(self, channels=1280, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                     nn.SiLU(inplace=True))
        self.attn3 = nn.Conv2d(hidden, channels, 1)
        self.attn5 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        f3 = self.branch3(x)
        f5 = self.branch5(x)
        z = self.reduce(self.pool(f3 + f5))
        logits = torch.stack((self.attn3(z), self.attn5(z)), dim=1)
        weights = torch.softmax(logits, dim=1)
        return weights[:, 0] * f3 + weights[:, 1] * f5

def build_model(name, pretrained=True):
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    elif name in ("efficientnet_v2_s", "efficientnet_v2_s_sk"):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        if name == "efficientnet_v2_s_sk":
            # Insert SK after the convolutional feature extractor and before global pooling.
            model.avgpool = nn.Sequential(SKAttention(1280), nn.AdaptiveAvgPool2d(1))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model

def prepare_real_dataset(cache_dir, output_dir, split_seed=2026):
    """Download source dataset, decode/validate images, remove exact duplicates,
    exclude hash collisions with conflicting labels, then make stratified 70/15/15 splits."""
    from datasets import load_dataset
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(SOURCE_DATASET, split=SOURCE_SPLIT)
    rows = []
    invalid_rows = []
    for i in range(len(ds)):
        try:
            row = ds[i]
            src_class = str(row["class"])
            if src_class not in SOURCE_TO_CLASS:
                invalid_rows.append({"source_index": i, "reason": "unknown_class", "source_class": src_class})
                continue
            image = row["image"]
            if image is None:
                raise ValueError("empty image")
            image = image.convert("RGB")
            if image.width < 16 or image.height < 16:
                raise ValueError("image dimensions below 16 pixels")
            digest = hashlib.sha256(
                str(image.size).encode("utf-8") + image.tobytes()
            ).hexdigest()
            filename = f"img_{i:05d}.jpg"
            image.save(image_dir / filename, format="JPEG", quality=95, optimize=True)
            rows.append({
                "source_index": i, "source_class": src_class,
                "subclass": str(row.get("subclass", "")),
                "label": SOURCE_TO_CLASS[src_class],
                "sha256_rgb_pixels": digest, "file_name": filename,
            })
        except Exception as exc:
            invalid_rows.append({"source_index": i, "reason": f"decode_error:{type(exc).__name__}"})

    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["sha256_rgb_pixels"]].append(r)
    clean = []
    exact_duplicate_count = 0
    conflicting_hash_count = 0
    for digest, group in by_hash.items():
        label_set = {r["label"] for r in group}
        if len(label_set) > 1:
            conflicting_hash_count += len(group)
            continue
        clean.append(group[0])
        exact_duplicate_count += max(0, len(group) - 1)

    if not clean:
        raise RuntimeError("No valid records remain after image cleaning.")
    labels = [r["label"] for r in clean]
    indices = np.arange(len(clean))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=split_seed, stratify=labels)
    trainval_labels = [labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=(0.15 / 0.85), random_state=split_seed,
        stratify=trainval_labels)

    split_for_index = {}
    for i in train_idx: split_for_index[int(i)] = "train"
    for i in val_idx: split_for_index[int(i)] = "val"
    for i in test_idx: split_for_index[int(i)] = "test"
    for i, r in enumerate(clean):
        r["split"] = split_for_index[i]
        r["relative_path"] = f"images/{r['file_name']}"

    manifest = pd.DataFrame(clean)
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    split_counts = {}
    class_counts = {}
    for split in ("train", "val", "test"):
        subset = manifest[manifest["split"] == split]
        split_counts[split] = int(len(subset))
        class_counts[split] = {c: int((subset["label"] == c).sum()) for c in CLASSES}
    audit = {
        "source_dataset": SOURCE_DATASET, "source_split": SOURCE_SPLIT,
        "source_rows": int(len(ds)), "decoded_valid_rows_before_dedup": int(len(rows)),
        "invalid_or_unrecognized_rows": int(len(invalid_rows)),
        "exact_duplicate_rows_removed": int(exact_duplicate_count),
        "rows_excluded_due_to_same_pixels_with_conflicting_labels": int(conflicting_hash_count),
        "final_unique_images": int(len(manifest)),
        "classes_in_model_order": CLASSES, "source_to_model_label_mapping": SOURCE_TO_CLASS,
        "split_seed": int(split_seed), "split_ratio_approx": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_counts": split_counts, "class_counts_by_split": class_counts,
        "cleaning_rules": [
            "Use the cleaned split described by the dataset maintainer as real-world photos.",
            "Reject unreadable images, unknown labels, and images with either dimension below 16 pixels.",
            "Convert images to RGB and compute SHA-256 over dimensions plus decoded RGB pixels.",
            "Remove exact duplicate pixel arrays; exclude all copies if identical pixels carry conflicting labels.",
            "Use a fixed, stratified 70/15/15 split; keep the test set untouched during model selection."
        ],
        "limitations": [
            "Dataset categories are the source maintainer's four classes; mapping Organic to kitchen and Non-Recyclable to other is a modeling convention, not a guarantee of full equivalence with every local municipal sorting rule.",
            "Exact pixel hashing does not detect visually similar images that have been resized or recompressed.",
            "The dataset card asks users to review the original Kaggle dataset terms before redistribution."
        ],
        "invalid_row_examples": invalid_rows[:50],
    }
    (output_dir / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest, audit, image_dir

class WasteDataset(Dataset):
    def __init__(self, records, image_root, transform):
        self.records = records.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records.iloc[idx]
        path = self.image_root / str(r["file_name"])
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image), CLASS_TO_ID[str(r["label"])], idx

def make_transforms(image_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_tf = transforms.Compose([
        transforms.Resize(image_size + 24), transforms.CenterCrop(image_size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return train_tf, eval_tf

def loader_for(records, image_root, transform, batch_size, shuffle, seed, workers):
    ds = WasteDataset(records, image_root, transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=False, generator=generator,
                      persistent_workers=(workers > 0))

def score_model(model, loader, device, class_weights=None):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    losses = []
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()))
            probs = torch.softmax(logits, dim=1)
            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            all_prob.extend(probs.cpu().numpy().tolist())
    p, r, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
        "y_true": all_true, "y_pred": all_pred, "probabilities": all_prob,
    }

def plot_history(history, path, title):
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(h["epoch"], h["train_loss"], label="Train loss")
    axes[0].plot(h["epoch"], h["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(h["epoch"], h["val_accuracy"], label="Validation accuracy")
    axes[1].plot(h["epoch"], h["val_macro_f1"], label="Validation macro-F1")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def benchmark_single_image(model, image_tensor, device, warmup=20, repeats=60):
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(x)
            samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "device": str(device), "input_shape": list(x.shape), "warmup_runs": warmup,
        "measured_runs": repeats, "mean_ms": float(np.mean(samples)),
        "std_ms": float(np.std(samples, ddof=1)), "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
    }

def run_one_seed(model_name, seed, manifest, image_root, out_dir, args):
    seed_everything(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_records = manifest[manifest["split"] == "train"].copy()
    val_records = manifest[manifest["split"] == "val"].copy()
    test_records = manifest[manifest["split"] == "test"].copy()
    train_tf, eval_tf = make_transforms(args.image_size)
    train_loader = loader_for(train_records, image_root, train_tf, args.batch_size, True, seed, args.workers)
    val_loader = loader_for(val_records, image_root, eval_tf, args.batch_size, False, seed, args.workers)
    test_loader = loader_for(test_records, image_root, eval_tf, args.batch_size, False, seed, args.workers)
    device = torch.device("cpu")
    model = build_model(model_name, pretrained=not args.no_pretrained).to(device)

    train_ids = [CLASS_TO_ID[x] for x in train_records["label"].tolist()]
    counts = np.bincount(train_ids, minlength=len(CLASSES))
    weights_np = len(train_ids) / (len(CLASSES) * np.maximum(counts, 1))
    class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses, train_true, train_pred = [], [], []
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_true.extend(y.detach().cpu().numpy().tolist())
            train_pred.extend(logits.detach().argmax(1).cpu().numpy().tolist())

        train_acc = float(accuracy_score(train_true, train_pred))
        val = score_model(model, val_loader, device, class_weights)
        scheduler.step(val["macro_f1"])
        record = {
            "epoch": epoch, "train_loss": float(np.mean(train_losses)),
            "train_accuracy": train_acc, "val_loss": val["loss"],
            "val_accuracy": val["accuracy"], "val_macro_precision": val["macro_precision"],
            "val_macro_recall": val["macro_recall"], "val_macro_f1": val["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps({"model": model_name, "seed": seed, **record}), flush=True)
        if val["macro_f1"] > best_score + 1e-6:
            best_score, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            torch.save({
                "model_name": model_name, "seed": seed, "state_dict": model.state_dict(),
                "classes": CLASSES, "image_size": args.image_size,
                "source_dataset": SOURCE_DATASET, "source_split": SOURCE_SPLIT,
                "best_epoch": best_epoch, "best_val_macro_f1": best_score,
                "source_to_model_label_mapping": SOURCE_TO_CLASS,
            }, best_path)
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping: {model_name} seed={seed}, epoch={epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    plot_history(history, out_dir / "training_curve.png", f"{model_name} | seed {seed}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test = score_model(model, test_loader, device, class_weights)
    cm = confusion_matrix(test["y_true"], test["y_pred"], labels=list(range(len(CLASSES))))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "confusion_matrix.csv")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False, xticks_rotation=30)
    ax.set_title(f"{model_name} | seed {seed} | held-out test set")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=160); plt.close(fig)

    report = classification_report(test["y_true"], test["y_pred"], labels=list(range(len(CLASSES))),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
    pred_records = test_records.reset_index(drop=True)
    pred_df = pd.DataFrame({
        "source_index": pred_records["source_index"].tolist(),
        "file_name": pred_records["file_name"].tolist(),
        "subclass": pred_records["subclass"].tolist(),
        "true_label": [CLASSES[i] for i in test["y_true"]],
        "predicted_label": [CLASSES[i] for i in test["y_pred"]],
        "confidence": [float(max(row)) for row in test["probabilities"]],
        "correct": [a == b for a, b in zip(test["y_true"], test["y_pred"])],
        "sha256_rgb_pixels": pred_records["sha256_rgb_pixels"].tolist(),
    })
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    pred_df[pred_df["correct"] == False].to_csv(out_dir / "misclassified_samples.csv", index=False)

    sample_img = next(iter(test_loader))[0][0]
    latency = benchmark_single_image(model, sample_img, device)
    metrics = {
        "model": model_name, "seed": seed, "source_dataset": SOURCE_DATASET,
        "source_split": SOURCE_SPLIT, "dataset_total_after_cleaning": int(len(manifest)),
        "train_n": int(len(train_records)), "val_n": int(len(val_records)), "test_n": int(len(test_records)),
        "class_order": CLASSES, "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_score), "test_loss": test["loss"],
        "accuracy": test["accuracy"], "macro_precision": test["macro_precision"],
        "macro_recall": test["macro_recall"], "macro_f1": test["macro_f1"],
        "test_classification_report": report, "inference_latency": latency,
        "checkpoint_file": "best.pt",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics, best_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["resnet18", "mobilenet_v3_small", "efficientnet_v2_s", "efficientnet_v2_s_sk"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--cache-dir", default="real_dataset_cache")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    output_root = Path(args.output_dir) / args.model
    output_root.mkdir(parents=True, exist_ok=True)
    manifest, audit, image_root = prepare_real_dataset(args.cache_dir, output_root, split_seed=2026)
    print("DATASET_AUDIT=" + json.dumps(audit, ensure_ascii=False), flush=True)
    all_metrics, checkpoints = [], []
    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        metrics, checkpoint = run_one_seed(args.model, seed, manifest, image_root, seed_dir, args)
        all_metrics.append(metrics)
        checkpoints.append((metrics["best_val_macro_f1"], checkpoint, seed))
    pd.DataFrame([{
        "model": m["model"], "seed": m["seed"], "best_epoch": m["best_epoch"],
        "test_accuracy": m["accuracy"], "test_macro_precision": m["macro_precision"],
        "test_macro_recall": m["macro_recall"], "test_macro_f1": m["macro_f1"],
        "latency_mean_ms": m["inference_latency"]["mean_ms"],
        "latency_std_ms": m["inference_latency"]["std_ms"],
    } for m in all_metrics]).to_csv(output_root / "model_seed_metrics.csv", index=False)

    # Keep one deployable checkpoint per architecture, chosen only by validation macro-F1.
    _, chosen_path, chosen_seed = max(checkpoints, key=lambda x: x[0])
    shutil.copy2(chosen_path, output_root / "best_model.pt")
    (output_root / "selected_checkpoint.json").write_text(json.dumps({
        "model": args.model, "selected_seed_by_validation_macro_f1": chosen_seed,
        "selection_rule": "Highest validation macro-F1 among the three fixed seeds; test metrics were not used for selection."
    }, indent=2), encoding="utf-8")
    # Avoid retaining three large duplicate checkpoints in the uploaded artifact.
    for _, path, seed in checkpoints:
        if path.exists():
            path.unlink()
    print(f"MODEL_DONE={args.model} seeds={args.seeds} selected_seed={chosen_seed}", flush=True)

if __name__ == "__main__":
    main()
