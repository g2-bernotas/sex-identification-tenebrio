"""
5-fold cross-validation over the full model-size x resolution grid, using stock
Ultralytics augmentation.

Per fold:
    test  = the held-out fold        (never trained on, never used to pick best.pt)
    val   = 10% of the remainder     (used only for checkpoint selection)
    train = the rest

Predictions from all folds are pooled per config, so each cell of the results
table is computed over every image in the dataset exactly once.

Expected input layout (one folder per class, images directly inside):

    SRC/
      female/*.bmp
      male/*.bmp
"""

import csv
import gc
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SRC = Path()    # PATH_TO_YOUR_DATA
WORK = Path()   # PATH_TO_WORKDIR, deleted after use
PROJECT = ""    # PATH_TO_YOUR_PROJECT

CLASSES = ["female", "male"]     # sorted folder order: female = 0, male = 1
K = 5
MAX_FOLDS = None                 # None = all K folds; set to 1 for a test
EPOCHS = 200
PATIENCE = 100
BATCH = 16
SEED = 0
VAL_FRACTION = 0.1               # slice off the remainder, for best.pt selection only

# Cheapest cells run first so partial results arrive early.
NAMES = ["nano", "small", "medium", "large", "x-large"]
SHORT = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
RESOLUTIONS = [64, 128, 256, 512, 640]

FOLD_TIMES = []                  # per-fold minutes, for the running ETA

import torchvision.transforms as T

DEGREES = 180              # ±180 covers the full 360°
TRANSLATE = (0.1, 0.1)     # fraction of width/height

def add_geometry(trainer):
    """ Ultralytics does not provide rotation/translation by default. 
    Insert rotation + translation ahead of the stock classification chain.
    Runs on PIL images before RandomResizedCrop, so the resize absorbs any
    resampling artefacts. fill=0 matches the segmented black background."""
    tf = trainer.train_loader.dataset.torch_transforms
    tf.transforms.insert(0, T.RandomAffine(degrees=DEGREES, translate=TRANSLATE, fill=0))
    print("train transforms:", tf)      # copy this into the Methods

# ----------------------------------------------------------------------------
# Fold construction
# ----------------------------------------------------------------------------

def collect(src: Path):
    files, labels = [], []
    for idx, cls in enumerate(CLASSES):
        for path in sorted((src / cls).iterdir()):
            if path.is_file():
                files.append(path)
                labels.append(idx)
    return np.array(files), np.array(labels)


def build_fold(root: Path, files, labels, train_idx, val_idx, test_idx):
    if root.exists():
        shutil.rmtree(root)
    for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in idx:
            dest = root / split / CLASSES[labels[i]]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(files[i], dest / files[i].name)


def append_csv(path: Path, rows):
    """Append rows, writing a header only if the file is new."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def done_configs(folds_csv: Path):
    """(model, resolution) pairs that already have all their folds on disk, so an
    interrupted sweep can be restarted without redoing finished cells."""
    if not folds_csv.exists():
        return set()
    seen = {}
    with open(folds_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["model"], int(row["resolution"]))
            seen[key] = seen.get(key, 0) + 1
    target = MAX_FOLDS or K
    return {key for key, n in seen.items() if n >= target}


# ----------------------------------------------------------------------------
# Inference on the held-out fold
# ----------------------------------------------------------------------------

def predict_fold(model: YOLO, paths, res: int, chunk=16):
    """Stock model.predict(), so the test images go through exactly the same
    transform Ultralytics uses at val time."""
    preds = []
    for i in range(0, len(paths), chunk):
        batch = [str(p) for p in paths[i:i + chunk]]
        for r in model.predict(batch, imgsz=res, verbose=False):
            preds.append(int(r.probs.top1))
    return preds


# ----------------------------------------------------------------------------
# One config = one model size at one resolution, cross-validated
# ----------------------------------------------------------------------------

def run_config(model_short: str, model_name: str, res: int, files, labels):
    folds_csv = Path(PROJECT) / "folds.csv"
    preds_csv = Path(PROJECT) / "predictions.csv"

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    pooled_true, pooled_pred, fold_acc = [], [], []

    for fold, (rest_idx, test_idx) in enumerate(skf.split(files, labels)):
        if MAX_FOLDS is not None and fold >= MAX_FOLDS:
            break
        fold_start = time.time()
        train_idx, val_idx = train_test_split(
            rest_idx, test_size=VAL_FRACTION, stratify=labels[rest_idx], random_state=SEED
        )

        tag = f"pupae-back-side-{res}px-{model_name}-f{fold}"
        root = WORK / tag
        build_fold(root, files, labels, train_idx, val_idx, test_idx)
        print(f"\n=== {tag}: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} ===")

        # a fresh model every fold — reusing one instance would carry weights over
        model = YOLO(f"{model_short}-cls.pt")
        model.add_callback("on_train_start", add_geometry)

        model.train(
            data=str(root),
            imgsz=res,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH,
            seed=SEED,
            project=PROJECT,
            name=tag,
            exist_ok=True,
        )

        # train() reloads best.pt into `model`, so this scores the selected weights
        fold_pred = predict_fold(model, files[test_idx], res)
        fold_true = [int(v) for v in labels[test_idx]]
        pooled_pred.extend(fold_pred)
        pooled_true.extend(fold_true)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(root, ignore_errors=True)

        # --- persist before anything else, so a crash costs at most one fold ---
        correct = sum(int(a == b) for a, b in zip(fold_true, fold_pred))
        acc = correct / len(fold_true)
        fold_acc.append(acc)
        elapsed = (time.time() - fold_start) / 60

        append_csv(folds_csv, [dict(
            model=model_name, resolution=res, fold=fold,
            n_test=len(fold_true), correct=correct, accuracy=round(acc, 4),
            macro_f1=round(f1_score(fold_true, fold_pred, average="macro"), 4),
            minutes=round(elapsed, 2),
            weights=str(Path(PROJECT) / tag / "weights" / "best.pt"),
        )])
        append_csv(preds_csv, [
            dict(model=model_name, resolution=res, fold=fold,
                 file=Path(p).name, true=CLASSES[t], pred=CLASSES[q])
            for p, t, q in zip(files[test_idx], fold_true, fold_pred)
        ])

        FOLD_TIMES.append(elapsed)
        done = len(FOLD_TIMES)
        total = len(NAMES) * len(RESOLUTIONS) * (MAX_FOLDS or K)
        avg = sum(FOLD_TIMES) / done
        print(f"[fold {fold}] {correct}/{len(fold_true)} = {acc:.3f} | {elapsed:.1f} min | "
              f"{done}/{total} folds this session | ~{avg * (total - done) / 60:.1f} h left")

    return np.array(pooled_true), np.array(pooled_pred), fold_acc


def report(model_name, res, y_true, y_pred, fold_acc):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1])
    correct = int((y_true == y_pred).sum())
    total = len(y_true)

    print(f"\n=== {model_name} @ {res}px — pooled over {MAX_FOLDS or K} fold(s) ===")
    print(f"macro F1 : {macro_f1:.3f}")
    print("per-class: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(CLASSES, per_class)))
    print(f"accuracy : {correct}/{total} = {correct / total:.3f}")
    print(f"fold acc : {np.mean(fold_acc):.3f} +- {np.std(fold_acc):.3f}")
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    return dict(model=model_name, resolution=res, epochs=EPOCHS, patience=PATIENCE,
                macro_f1=round(macro_f1, 4),
                f1_female=round(per_class[0], 4), f1_male=round(per_class[1], 4),
                correct=correct, total=total,
                fold_mean=round(float(np.mean(fold_acc)), 4),
                fold_sd=round(float(np.std(fold_acc)), 4))


def main():
    files, labels = collect(SRC)
    counts = {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)}
    print(f"{len(files)} images: {counts}")

    Path(PROJECT).mkdir(parents=True, exist_ok=True)
    folds_csv = Path(PROJECT) / "folds.csv"
    summary_csv = Path(PROJECT) / "summary.csv"
    finished = done_configs(folds_csv)
    if finished:
        print(f"resuming — skipping {len(finished)} completed config(s)")

    for name, sh in zip(NAMES, SHORT):
        for res in RESOLUTIONS:
            if (name, res) in finished:
                continue
            y_true, y_pred, fold_acc = run_config(sh, name, res, files, labels)
            if len(y_true) == 0:
                continue
            append_csv(summary_csv, [report(name, res, y_true, y_pred, fold_acc)])

    print(f"\nSummary written to {summary_csv}")


if __name__ == "__main__":
    main()