# Automated Sex Identification of *Tenebrio molitor* (Pupal & Adult Stages)

This repository contains the code used in the study **Automated Sex Identification of Yellow Mealworm (*Tenebrio molitor*) at Pupal and Adult Stages**. The project uses **Ultralytics YOLO** models for classification and includes **EigenCAM** visualisation for model interpretability. The project employs k-fold to leverage small insect dataset. 

The repository consists of a small set of standalone Python scripts for:
- Training and evaluating YOLOv8 classification models on pupal and adult images
- Generating evaluation metrics (macro-F1, confusion matrices (per model))
- The code will generate summary.csv, predictions.csv, and folds.csv per experiment  
- Producing EigenCAM visualisations to highlight relevant regions to model
- Reproducible analysis used in the manuscript

All scripts are self-contained and rely on `requirements.txt` for environment setup.

# Data availability
Data and model weights are available [here](https://doi.org/10.5281/zenodo.21996433). 

# Installation

```bash
pip install -r requirements.txt
```

# References

Ultralytics YOLO:  
https://github.com/ultralytics/ultralytics

EigenCAM (adapted from):  
https://github.com/rigvedrs/YOLO-26-CAM
