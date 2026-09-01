"""
Assemble the per-insect 5x5 Eigen-CAM grid.

    rows    = resolution, 64px (top) -> 640px (bottom)
    columns = model size, nano (left) -> x-large (right)

Reads the panels written by eigencam_generate.py and produces, per insect:

    grids/layer-2/<stem>_grid.pdf      publication figure, no annotation
    grids/layer-2/<stem>_grid_preview.png   same but annotated with the
                                            prediction in each cell, for
                                            quickly picking the best insect

EDITING ONE OF THE 25 PANELS
----------------------------
The PDF is a vector container: each panel is a separate embedded image object,
so in Illustrator / Inkscape / Affinity you can click a single cell, delete it
and place a replacement without touching the other 24. Text (row and column
labels) is written as editable TrueType, not outlines, via pdf.fonttype = 42.

The individual panel PNGs that back each cell are already on disk at
DISPLAY_SIZE px (1024 by default) under
    <OUT>/layer-2/<stem>/<model>_<res>px.png
so you can re-edit or replace any single cell directly.

Set FORMATS = ["pdf", "eps", "svg"] if you also want those. EPS carries no
transparency and is legacy, but Illustrator opens it fine.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# keep text editable in the vector outputs rather than converting to paths
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# ----------------------------------------------------------------------------
# Configuration — must match eigencam_generate.py
# ----------------------------------------------------------------------------
# PROJECT = Path(r"C:\Users\Gytis\yolov7\runs\v8-kfold")

# PROJECT = Path(r"C:\Users\cvmbrl\yolov8\runs\pupae-back-kfold") # done

# PROJECT = Path(r"C:\Users\cvmbrl\yolov8\runs\pupae-back-side-kfold")
# PROJECT = Path(r"C:\Users\cvmbrl\yolov8\runs\pupae-side-kfold")
# PROJECT = Path(r"C:\Users\cvmbrl\yolov8\runs\adult-kfold")

PROJECT = Path(r"C:\Users\cvmbrl\yolov8\runs\pupae-back-kfold")


OUT = PROJECT / "eigencam"
GRIDS = OUT / "grids"

MODELS = ["nano", "small", "medium", "large", "x-large"]
RESOLUTIONS = [64, 128, 256, 512, 640]
LAYERS = (-3, -2)

FORMATS = ["pdf"]        # add "eps" / "svg" if you want them too
PANEL_INCHES = 1.7       # per cell; 5 cells -> ~8.5 in wide, journal page width
PREVIEW_DPI = 130
ONLY_STEMS = []          # empty = every insect found under OUT/layer*/


# ----------------------------------------------------------------------------

def load_index():
    """(stem, model, res, layer) -> row from cam_index.csv, for the preview labels."""
    idx_csv = OUT / "cam_index.csv"
    if not idx_csv.exists():
        return {}
    table = {}
    with open(idx_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["stem"], row["model"], int(row["resolution"]), int(row["layer"]))
            table[key] = row
    return table


def discover_stems(layer):
    root = OUT / f"layer{layer}"
    if not root.exists():
        return []
    stems = sorted(d.name for d in root.iterdir() if d.is_dir())
    if ONLY_STEMS:
        stems = [s for s in stems if s in ONLY_STEMS]
    return stems


def build_grid(stem, layer, index, annotate):
    root = OUT / f"layer{layer}" / stem

    fig_w = PANEL_INCHES * len(MODELS)
    fig_h = PANEL_INCHES * len(RESOLUTIONS)
    fig, axes = plt.subplots(len(RESOLUTIONS), len(MODELS),
                             figsize=(fig_w, fig_h))

    # margins reserved in inches, so the layout holds at any PANEL_INCHES
    left_in = 0.55                          # row labels
    top_in = 0.35 + (0.32 if annotate else 0.0)   # column titles (+ suptitle)
    fig.subplots_adjust(left=left_in / fig_w, right=0.995,
                        top=1 - top_in / fig_h, bottom=0.005,
                        wspace=0.02, hspace=0.02)

    missing = 0
    for r, res in enumerate(RESOLUTIONS):
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)

            panel = root / f"{model}_{res}px.png"
            if panel.exists():
                ax.imshow(mpimg.imread(str(panel)))
            else:
                missing += 1
                ax.set_facecolor("0.85")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        fontsize=7, color="0.4", transform=ax.transAxes)

            if annotate:
                row = index.get((stem, model, res, layer))
                if row:
                    ok = row["correct"] == "1"
                    ax.text(0.03, 0.97,
                            f"{row['pred'][0].upper()} {float(row['prob']):.2f}",
                            transform=ax.transAxes, ha="left", va="top",
                            fontsize=7, color="white",
                            bbox=dict(facecolor="green" if ok else "red",
                                      alpha=0.75, pad=1.2, edgecolor="none"))

            if r == 0:
                ax.set_title(model, fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(f"{res}x{res}", fontsize=11, labelpad=6)

    if annotate:
        row = next((index[k] for k in index
                    if k[0] == stem and k[3] == layer), None)
        if row:
            fig.suptitle(f"{stem} — true: {row['true']} (fold {row['fold']}, "
                         f"layer {layer})", fontsize=10,
                         y=1 - 0.14 / fig_h)

    return fig, missing


def main():
    index = load_index()
    total = 0

    for layer in LAYERS:
        stems = discover_stems(layer)
        if not stems:
            print(f"[skip] no panels found for layer {layer}")
            continue
        dst_dir = GRIDS / f"layer{layer}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"layer {layer}: {len(stems)} insects")

        for stem in stems:
            # clean vector figure for the paper
            fig, missing = build_grid(stem, layer, index, annotate=False)
            for ext in FORMATS:
                fig.savefig(dst_dir / f"{stem}_grid.{ext}", format=ext,
                            bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

            # annotated raster for browsing
            fig, _ = build_grid(stem, layer, index, annotate=True)
            fig.savefig(dst_dir / f"{stem}_grid_preview.png", dpi=PREVIEW_DPI,
                        bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

            total += 1
            if missing:
                print(f"  {stem}: {missing}/25 panels missing")

    print(f"\n{total} grids written under {GRIDS}")
    print("browse the *_grid_preview.png files, then edit the matching "
          "*_grid.pdf for the one you pick")


if __name__ == "__main__":
    main()