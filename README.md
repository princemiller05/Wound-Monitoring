# AI-Assisted Remote Wound Monitoring System for Telehealth Applications

An end-to-end pipeline for analyzing **Diabetic Foot Ulcers (DFUs)** from a single photograph. It tells you how big the wound is, what kind of tissue it's made of, and whether it's likely to heal.

Built as part of a group project for remote telehealth applications. Three modules — **wound segmentation**, **tissue classification**, and **healing prediction** — now chained into one clean pipeline you can run with a single command.

---

## What It Does

Give it a wound photo. Get back:

1. **A segmentation mask** — exactly where the wound is (pixel level)
2. **A tissue breakdown** — how much is granulation (healthy), slough, or necrosis
3. **A healing prediction** — probability that this wound will heal, with reasoning

All the intermediate images (masks, overlays, tissue maps, trend plots) get saved to disk so you can see exactly what the pipeline is doing at each step.

---

## Pipeline Overview

```
   Wound Photo
       │
       ▼
┌──────────────────────┐
│  1. Segmentation     │   YOLO finds the wound → MedSAM outlines it precisely
│  (Prince)            │   Outputs: mask, overlay, wound area (px)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  2. Tissue Classify  │   ResNet18 slides across wound region,
│  (Subham)            │   classifying 64x64 patches as granulation / slough / necrosis
│                      │   Outputs: % of each tissue type, colored heatmap
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  3. Healing Predict  │   Simulates 21-day progression from Day-0 mask,
│  (Varsha)            │   XGBoost predicts healing probability
│                      │   Outputs: probability, label, area/tissue trend plots
└──────────────────────┘
       │
       ▼
  PipelineResult
```

---

## Repo Structure

```
dfu_pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── run_demo.py                   # one-command demo
│
├── data/
│   └── sample_inputs/            # drop wound images here
│
├── models/                       # trained weights go here (not in git, see below)
│   ├── segmentation/
│   │   ├── best.pt               # YOLO
│   │   └── medsam_vit_b.pth      # MedSAM
│   ├── tissue/
│   │   └── tissue_model.pth      # ResNet18
│   └── healing/
│       └── xgb_healing.json      # XGBoost
│
├── outputs/                      # generated at runtime (gitignored)
│   ├── masks_pred/               # binary wound masks
│   ├── overlays/                 # contour drawn on original
│   ├── crops/                    # isolated wound, background removed
│   ├── tissue_preds/             # colored tissue heatmaps
│   └── healing/                  # trend plots + CSVs
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py                 # all paths and thresholds
│   ├── schemas.py                # output dataclasses
│   ├── orchestrator.py           # the top-level DFUPipeline class
│   │
│   ├── segmentation/             # Prince's module
│   │   ├── model.py              # loads YOLO + MedSAM
│   │   ├── preprocess.py         # image loading / resizing
│   │   ├── infer.py              # detection + segmentation
│   │   └── utils.py              # mask cleanup, overlays, area
│   │
│   ├── tissue/                   # Subham's module
│   │   ├── model.py              # loads ResNet18
│   │   ├── preprocess.py         # patch sampling + transforms
│   │   ├── infer.py              # sliding-window classification
│   │   └── utils.py              # tissue map overlay, percentages
│   │
│   └── healing/                  # Varsha's module
│       ├── model.py              # loads XGBoost
│       ├── synthetic_progression.py  # simulates day 7/14/21 masks
│       ├── features.py           # longitudinal dataset + summary features
│       ├── rule_baseline.py      # 30% area reduction rule
│       ├── infer.py              # main prediction function
│       └── utils.py              # trend plots
│
├── notebooks/                    # original notebooks, kept for reference
│   ├── prince_segmentation.ipynb
│   ├── subham_tissue.ipynb
│   └── varsha_healing.ipynb
│
├── scripts/
│   └── train_healing_model.py    # regenerate xgb_healing.json
│
└── tests/
    ├── test_segmentation.py
    ├── test_tissue.py
    └── test_healing.py
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/dfu-pipeline.git
cd dfu-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Drop in the model weights

The weights aren't committed to the repo (they're too big for git). Place them here:

| File                  | Goes in                     |
|-----------------------|-----------------------------|
| `best.pt`             | `models/segmentation/`      |
| `medsam_vit_b.pth`    | `models/segmentation/`      |
| `tissue_model.pth`    | `models/tissue/`            |
| `xgb_healing.json`    | `models/healing/`           |

If you don't have `xgb_healing.json`, regenerate it:

```bash
python scripts/train_healing_model.py
```

That trains XGBoost on synthetic data and saves the model. Takes a few seconds.

---

## Running It

### Quickest way

Put a wound image in `data/sample_inputs/`, then:

```bash
python run_demo.py
```

### On a specific image

```bash
python run_demo.py --image data/sample_inputs/woundtst.jpg --case CASE_001
```

### From Python

```python
from pipeline.orchestrator import DFUPipeline

# Load all models once — this takes a few seconds
pipe = DFUPipeline()

# Then run on as many images as you want — each run is fast
result = pipe.run("data/sample_inputs/woundtst.jpg", case_id="CASE_001")

# Access structured results
print(result.segmentation.area_px)          # e.g. 39246
print(result.tissue.granulation_pct)        # e.g. 100.0
print(result.healing.healing_probability)   # e.g. 0.9856
print(result.healing.predicted_label)       # "healing" or "non_healing"
```

### CPU-only mode

If you don't have a GPU or CUDA is giving you grief:

```python
pipe = DFUPipeline(device="cpu")
```

### Batch processing many images

```python
from pathlib import Path

pipe = DFUPipeline()

for i, img_path in enumerate(Path("data/sample_inputs").glob("*.jpg")):
    pipe.run(str(img_path), case_id=f"CASE_{i:03d}")
```

---

## Naming Convention

We use a consistent naming scheme across all modules so results can be joined later:

| Field       | Format           | Example          |
|-------------|------------------|------------------|
| `case_id`   | `CASE_NNN`       | `CASE_001`       |
| `image_id`  | `CASE_NNN_DAYN`  | `CASE_001_DAY0`  |
| `wound_id`  | `WN`             | `W1`             |
| `visit_day` | integer          | `0, 7, 14, 21`   |

---

## Output Schemas

Each module returns a typed dataclass (see `pipeline/schemas.py`):

- `SegmentationResult` — `mask_path`, `overlay_path`, `crop_path`, `area_px`, `bbox`, `yolo_conf`
- `TissueResult` — `granulation_pct`, `slough_pct`, `necrosis_pct`, `tissue_map_path`
- `HealingResult` — `healing_probability`, `predicted_label`, `rule_label`, `key_factors`
- `PipelineResult` — all three combined

---

## Tests

```bash
cd dfu_pipeline
pytest tests/
```

Tests cover the pure-Python logic (feature extraction, rule baseline, mask postprocessing). The model-heavy pieces are tested end-to-end by running `run_demo.py` on a sample image.

---

## Credits

Built as a group project:

- **Prince Betuverma** — wound segmentation (YOLO + MedSAM), pipeline integration
- **Subham** — tissue classification (ResNet18)
- **Varsha** — healing prediction (XGBoost)

---

## Roadmap

- [x] Unify three notebooks into one pipeline
- [x] End-to-end local inference
- [ ] Real multi-visit longitudinal data support (currently uses synthetic)
- [ ] Deploy as a cloud service (REST API)
- [ ] Mobile app for patient-side photo capture
