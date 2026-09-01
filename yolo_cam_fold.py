"""
Eigen-CAM over the full YOLOv8-cls grid (5 model sizes x 5 resolutions),
using fold-aware checkpoint selection so every CAM is computed by a model
that never saw that image.

The Eigen-CAM maths is identical to yolo_cam / pytorch-grad-cam: hook the
target layer, reshape activations to (HW, C), mean-centre, SVD, project onto
the first right singular vector, min-max normalise, resize. Set
VERIFY_AGAINST_YOLO_CAM = True to assert that equivalence numerically against
your local YOLO-V8-CAM-main checkout.

Output layout
-------------
OUT/
  cam_index.csv                       one row per (image, model, res, layer)
  layer-2/<stem>/nano_64px.png ... x-large_640px.png
  layer-2/<stem>/input.png            padded original, no overlay
  layer-3/...

Run eigencam_grid.py afterwards to assemble the 5x5 figures.
"""

import csv
import gc
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Configuration  (paths lifted from the k-fold script)
# ----------------------------------------------------------------------------
SRC = Path("")              # YOUR_DATA_PATH
PROJECT = Path("")          # YOUR_PROJECT_PATH
OUT = PROJECT / "eigencam"

YOLO_CAM_PATH = r"./YOLO-V8-CAM-main"

CLASSES = ["female", "male"]          # sorted order: female = 0, male = 1
K = 5
SEED = 0

# ascending order — the grid figure uses exactly this order
MODELS = ["nano", "small", "medium", "large", "x-large"]
RESOLUTIONS = [64, 128, 256, 512, 640]

MODELS.reverse()
RESOLUTIONS.reverse()

# Must match the trained runs. Normalisation off: the segmented background is
# zero-valued, and the A/B test showed ImageNet norm collapses this model.
USE_IMAGENET_NORM = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Negative indices into model.model.model (the backbone Sequential).
# For yolov8n-cls that Sequential is 0..9 with 9 = Classify, so:
#   -2 -> layer 8, the last C2f  (the usual Eigen-CAM target)
#   -3 -> layer 7, the stride-32 Conv feeding it
# NOTE ON LOW RESOLUTION: both of these sit at stride 32, so at 64px the
# feature map is only 2x2 and the CAM is four blocks. At 640px it is 20x20.
# If the 64px row looks useless, add -5 (stride 16) to this tuple.
TARGET_LAYERS = (-3, -2)

# Panel size in pixels. All 25 panels are rendered at this size from the
# original crop so the insect is pixel-identical across the grid and only the
# heatmap coarseness changes with resolution. The CAM itself is always computed
# at the model's native resolution, then upsampled for display.
DISPLAY_SIZE = 1024

# Overlay style, kept from your previous script (COLORMAP_HOT, 0.6/0.6).
# 0.6 + 0.6 = 1.2 deliberately overshoots and clips the highlights; drop
# ALPHA_CAM to 0.4 if you want the insect texture to stay readable underneath.
COLORMAP = cv2.COLORMAP_HOT
ALPHA_IMG = 0.6
ALPHA_CAM = 0.6

ONLY_STEMS = []          
MAX_PER_CLASS = None     # e.g. 20 for a quick look

# One progress bar over the whole job instead of a line per fold. Set False for
# the old per-fold "[done] ..." logging, e.g. if you are piping to a file.
PROGRESS_BAR = True

VERIFY_AGAINST_YOLO_CAM = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
# Progress bar
#
# Deliberately dependency-free: recent Ultralytics ships its own TQDM wrapper
# and no longer pulls in the tqdm package, and the two have incompatible APIs.
# ----------------------------------------------------------------------------

class Progress:
    """Single-line bar with elapsed and ETA. Falls back to periodic lines when
    stdout is redirected to a file."""

    def __init__(self, total, desc="", width=32, enabled=True):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.enabled = enabled
        self.n = 0
        self.status = ""
        self.start = time.time()
        self.last_draw = 0.0
        self.last_pct = -1
        self.tty = sys.stdout.isatty()
        if self.enabled:
            self._draw(force=True)

    @staticmethod
    def _hms(seconds):
        seconds = int(max(seconds, 0))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _line(self):
        frac = self.n / self.total
        filled = int(self.width * frac)
        elapsed = time.time() - self.start
        eta = elapsed / self.n * (self.total - self.n) if self.n else 0
        rate = self.n / elapsed if elapsed > 0 else 0
        bar = "#" * filled + "-" * (self.width - filled)
        return (f"{self.desc} [{bar}] {frac * 100:5.1f}%  "
                f"{self.n}/{self.total}  "
                f"{self._hms(elapsed)}<{self._hms(eta)}  "
                f"{rate:.2f} img/s  {self.status}")

    def _draw(self, force=False):
        if not self.enabled:
            return
        now = time.time()
        if self.tty:
            if not force and now - self.last_draw < 0.2:
                return
            self.last_draw = now
            sys.stdout.write("\r\033[K" + self._line())
            sys.stdout.flush()
        else:
            pct = int(100 * self.n / self.total)
            if force or pct >= self.last_pct + 2:
                self.last_pct = pct
                print(self._line(), flush=True)

    def set_status(self, text):
        self.status = text

    def update(self, k=1):
        self.n += k
        self._draw()

    def write(self, msg):
        """Print a message without leaving a half-drawn bar behind it."""
        if self.enabled and self.tty:
            sys.stdout.write("\r\033[K")
        print(msg, flush=True)
        self._draw(force=True)

    def close(self):
        if self.enabled and self.tty:
            self._draw(force=True)
            sys.stdout.write("\n")
            sys.stdout.flush()


# ----------------------------------------------------------------------------
# Transforms — copied verbatim from the k-fold script so the CAM input matches
# what the checkpoint was validated on
# ----------------------------------------------------------------------------

class PadSquare:
    """Pad to square with black. The pad colour IS the segmented background."""

    def __call__(self, im):
        w, h = im.size
        m = max(w, h)
        out = Image.new(im.mode, (m, m), 0)
        out.paste(im, ((m - w) // 2, (m - h) // 2))
        return out

    def __repr__(self):
        return "PadSquare()"


def _tail():
    tail = [T.ToTensor()]
    if USE_IMAGENET_NORM:
        tail.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return tail


def val_tf(sz):
    return T.Compose([PadSquare(), T.Resize((sz, sz)), *_tail()])


def display_image(path: Path) -> np.ndarray:
    """Padded original at DISPLAY_SIZE, BGR uint8 — the backdrop for every panel."""
    im = Image.open(path).convert("RGB")
    im = PadSquare()(im)
    im = im.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# Eigen-CAM
# ----------------------------------------------------------------------------

def get_2d_projection(activation_batch: np.ndarray) -> np.ndarray:
    """First principal component of the activation map.

    Identical to yolo_cam.utils.svd_on_activations.get_2d_projection (and to
    jacobgil's pytorch-grad-cam). Mean-centring before the SVD matters: without
    it the projection frequently comes out sign-flipped.
    """
    activation_batch = np.asarray(activation_batch, dtype=np.float32).copy()
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped = activations.reshape(activations.shape[0], -1).transpose()  # (HW, C)
        reshaped = reshaped - reshaped.mean(axis=0)
        _, _, VT = np.linalg.svd(reshaped, full_matrices=True)
        projection = reshaped @ VT[0, :]
        projections.append(projection.reshape(activations.shape[1:]))
    return np.float32(projections)


def scale_cam_image(cam: np.ndarray, target_size=None) -> np.ndarray:
    """Per-image min-max to [0, 1], then resize. Matches yolo_cam.utils.image."""
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    return np.float32(result)


class ActivationGrabber:
    """Forward hooks on the target layers; one forward pass feeds every CAM."""

    def __init__(self, module_seq, layer_indices):
        self.acts = {}
        self.handles = []
        for idx in layer_indices:
            module = module_seq[idx]
            self.handles.append(
                module.register_forward_hook(self._make_hook(idx))
            )

    def _make_hook(self, idx):
        def hook(_module, _inp, out):
            if isinstance(out, (list, tuple)):
                out = out[0]
            self.acts[idx] = out.detach().cpu().numpy()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def overlay(bgr_display: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
    """Your previous look: HOT colormap, addWeighted(img 0.6, heat 0.6)."""
    cam_resized = scale_cam_image(
        grayscale_cam[None, ...], (bgr_display.shape[1], bgr_display.shape[0])
    )[0]
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), COLORMAP)
    return cv2.addWeighted(bgr_display, ALPHA_IMG, heat, ALPHA_CAM, 0)


# ----------------------------------------------------------------------------
# Fold reconstruction — must mirror collect() + StratifiedKFold in the k-fold script
# ----------------------------------------------------------------------------

def collect(src: Path):
    files, labels = [], []
    for idx, cls in enumerate(CLASSES):
        for path in sorted((src / cls).iterdir()):
            if path.is_file():
                files.append(path)
                labels.append(idx)
    return np.array(files), np.array(labels)


def fold_assignment(files, labels):
    """image index -> the fold that held it out."""
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    fold_of = np.full(len(files), -1, dtype=int)
    for fold, (_rest, test_idx) in enumerate(skf.split(files, labels)):
        fold_of[test_idx] = fold
    assert (fold_of >= 0).all(), "some image was never in a test split"
    return fold_of


def cross_check_with_predictions(files, fold_of):
    """If predictions.csv exists, confirm the reconstructed split matches it."""
    preds_csv = PROJECT / "predictions.csv"
    if not preds_csv.exists():
        print("[check] predictions.csv not found — skipping split cross-check")
        return
    by_name = {p.name: fold_of[i] for i, p in enumerate(files)}
    mismatches = 0
    seen = 0
    with open(preds_csv, newline="") as f:
        for row in csv.DictReader(f):
            name, fold = row["file"], int(row["fold"])
            if name not in by_name:
                continue
            seen += 1
            if by_name[name] != fold:
                mismatches += 1
                if mismatches <= 5:
                    print(f"[check] MISMATCH {name}: csv fold {fold}, "
                          f"reconstructed {by_name[name]}")
    if mismatches:
        raise SystemExit(
            f"[check] {mismatches}/{seen} rows disagree with the reconstructed "
            "split. The CAMs would not be held-out. Check CLASSES order, SEED "
            "and that SRC still holds exactly the images used for training."
        )
    print(f"[check] split matches predictions.csv across {seen} rows")


def ckpt_path(model_name: str, res: int, fold: int) -> Path:
    # tag = f"paper-adult-beetle-single-{res}px-{model_name}-model-f{fold}"
    # tag = f"paper-pupae-single-{res}px-{model_name}-model-f{fold}"
    tag = f"pupae-{res}px-{model_name}-f{fold}"

    return PROJECT / tag / "weights" / "best.pt"


# ----------------------------------------------------------------------------
# Optional equivalence check against your local YOLO-V8-CAM checkout
# ----------------------------------------------------------------------------

def verify_against_yolo_cam(acts):
    import sys
    sys.path.append(YOLO_CAM_PATH)
    from yolo_cam.utils.svd_on_activations import get_2d_projection as ref
    mine = get_2d_projection(acts)
    theirs = ref(acts.copy())
    delta = np.abs(mine - theirs).max()
    print(f"[verify] max |mine - yolo_cam| = {delta:.3e}")
    assert delta < 1e-4, "Eigen-CAM projection diverges from yolo_cam"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    files, labels = collect(SRC)
    print(f"{len(files)} images: "
          f"{ {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)} }")

    # panels are filed by stem, so a duplicate stem would silently overwrite
    stems = [p.stem for p in files]
    if len(set(stems)) != len(stems):
        dupes = sorted({s for s in stems if stems.count(s) > 1})
        raise SystemExit(f"duplicate filenames across classes: {dupes[:10]}")

    fold_of = fold_assignment(files, labels)
    cross_check_with_predictions(files, fold_of)

    # ---- select the images to visualise -------------------------------------
    keep = np.ones(len(files), dtype=bool)
    if ONLY_STEMS:
        wanted = set(ONLY_STEMS)
        keep &= np.array([p.stem in wanted for p in files])
    if MAX_PER_CLASS is not None:
        capped = np.zeros(len(files), dtype=bool)
        for c in range(len(CLASSES)):
            idx = np.where(keep & (labels == c))[0][:MAX_PER_CLASS]
            capped[idx] = True
        keep &= capped
    sel = np.where(keep)[0]
    print(f"visualising {len(sel)} images x {len(MODELS) * len(RESOLUTIONS)} "
          f"configs x {len(TARGET_LAYERS)} layers "
          f"= {len(sel) * len(MODELS) * len(RESOLUTIONS) * len(TARGET_LAYERS)} CAMs")

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- backdrops, written once per insect ---------------------------------
    # Deliberately NOT held in RAM: at DISPLAY_SIZE = 1024 each is ~3 MB, so
    # caching all 342 would cost ~1 GB. They are re-read from disk in the loop
    # below, which is trivial next to the forward pass.
    prep = Progress(len(sel), desc="backdrops", enabled=PROGRESS_BAR)
    for i in sel:
        stem = files[i].stem
        disp = display_image(files[i])
        for layer in TARGET_LAYERS:
            d = OUT / f"layer{layer}" / stem
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "input.png"), disp)
        prep.update()
    prep.close()
    backdrop_of = {files[i].stem:
                   OUT / f"layer{TARGET_LAYERS[0]}" / files[i].stem / "input.png"
                   for i in sel}

    index_rows = []
    warned = set()
    verified = not VERIFY_AGAINST_YOLO_CAM

    # ---- progress -----------------------------------------------------------
    total_units = len(sel) * len(MODELS) * len(RESOLUTIONS)
    bar = Progress(total_units, desc="eigen-cam", enabled=PROGRESS_BAR)
    log = bar.write   # warnings without tearing the bar apart

    # ---- 125 checkpoints: one load, then every selected image in that fold ---
    for model_name in MODELS:
        for res in RESOLUTIONS:
            tf = val_tf(res)
            for fold in range(K):
                members = [i for i in sel if fold_of[i] == fold]
                if not members:
                    continue
                bar.set_status(f"{model_name} {res}px f{fold}")
                ck = ckpt_path(model_name, res, fold)
                if not ck.exists():
                    log(f"[skip] missing checkpoint {ck}")
                    bar.update(len(members))
                    continue

                yolo = YOLO(str(ck))
                net = yolo.model.float().eval().to(DEVICE)
                seq = net.model                      # the backbone Sequential
                n_out = getattr(net, "nc", None) or len(getattr(net, "names", {}))
                if n_out and n_out != len(CLASSES):
                    raise SystemExit(
                        f"{ck} has {n_out} output classes, expected "
                        f"{len(CLASSES)} — wrong checkpoint?"
                    )
                grab = ActivationGrabber(seq, TARGET_LAYERS)

                for i in members:
                    stem = files[i].stem
                    disp = cv2.imread(str(backdrop_of[stem]))
                    im = Image.open(files[i]).convert("RGB")
                    x = tf(im).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = net(x)
                    # Classify.forward returns (softmax, logits) in eval mode
                    probs = (out[0] if isinstance(out, (list, tuple)) else out)
                    probs = probs.squeeze(0).cpu().numpy()
                    pred = int(probs.argmax())

                    for layer in TARGET_LAYERS:
                        acts = grab.acts[layer]      # (1, C, H, W)
                        if not verified:
                            verify_against_yolo_cam(acts)
                            verified = True
                        h, w = acts.shape[2], acts.shape[3]
                        if (h < 4 or w < 4) and (model_name, res, layer) not in warned:
                            warned.add((model_name, res, layer))
                            log(f"[note] {model_name} @ {res}px layer {layer}: "
                                f"feature map is {h}x{w} — CAM will be blocky")
                        cam = get_2d_projection(acts)[0]
                        img = overlay(disp, cam)
                        dst = (OUT / f"layer{layer}" / stem /
                               f"{model_name}_{res}px.png")
                        cv2.imwrite(str(dst), img)

                        index_rows.append(dict(
                            stem=stem, file=files[i].name,
                            true=CLASSES[labels[i]], pred=CLASSES[pred],
                            prob=round(float(probs[pred]), 4),
                            correct=int(pred == labels[i]),
                            model=model_name, resolution=res, fold=fold,
                            layer=layer, feat_h=h, feat_w=w,
                            checkpoint=str(ck), panel=str(dst),
                        ))

                    bar.update()

                grab.close()
                del yolo, net, seq
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

    bar.close()

    if not index_rows:
        raise SystemExit(
            "no CAMs were produced — every checkpoint was missing. "
            "Run check_setup.py to see what is actually on disk."
        )

    idx_csv = OUT / "cam_index.csv"
    with open(idx_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"\n{len(index_rows)} CAMs written under {OUT}")
    print(f"index: {idx_csv}")
    print("next: run eigencam_grid.py")


if __name__ == "__main__":
    main()