# Project 2: Synthetic-to-Real Generalization for Chessboard Square and Board-State Classification

## Overview

This project explores **synthetic-to-real (sim-to-real) generalization** for chessboard perception.
We train a **per-square multi-class classifier** primarily on **synthetic chessboard images**
generated using Blender, and evaluate how well the trained model generalizes to **real chessboard images**.

The project examines:
- Zero-shot generalization (training on synthetic data only)
- Fine-tuning using a small amount of real data
- Joint training on synthetic and real data

The final output of the system is a **full chessboard state**, represented as a **FEN string**
and a reconstructed chessboard image, obtained from per-square classifications.

---

## Project Goals

- Train a per-square chessboard classifier using synthetic images
- Measure the domain gap between synthetic and real chessboard images
- Evaluate sim-to-real transfer effectiveness
- Compare zero-shot, fine-tuned, and combined training setups
- Produce a full board-state (FEN) prediction and a chessboard image from a single static image

---

## Repository Structure

- `blender/` – Synthetic data generation using Blender API.
- `preprocessing/` – Image preprocessing for synthetic and real data.
- `models/` – Training scripts for zero-shot, fine-tuned and combined models.
- `inference/` – Board-state prediction from a single RGB image.
- `demo.py` – End-to-end inference demo.

```text
chess-sim2real/
├── README.md
├── requirements.txt
├── demo.py                      
│
├── blender/
│   ├── chess_position_api_v2.py
│   ├── data_generator_from_csv.py
│   └── data_generator_random.py
│
├── preprocessing/
│   ├── crop_synthetic.py
│   ├── shift_and_pad.py
│   ├── split_to_squares.py
│   └── split_train_val.py
│
├── models/
│   ├── train_zero_shot.py
│   ├── train_fine_tuned.py
│   └── train_combined.py
│
├── inference/
│   └── predict_board.py
│
├── utils/
│   └── __init__.py
│
├── checkpoints/
│   ├── README.md
│   ├── zero_shot.pth
│   ├── fine_tuned.pth
│   └── combined.pth    
│
└── data/
    └── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/chess-sim2real.git
cd chess-sim2real
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```
**Note:**  
Synthetic data generation scripts rely on Blender's internal Python environment
(`bpy`, `mathutils`) and are not installed via pip.
These dependencies are **not required** for training or inference.

---

## Training
The training datasets (synthetic and real chessboard images) are not included in this
repository due to size constraints and course data distribution policies.
Pretrained model checkpoints are provided and are sufficient for reproducing the
reported results and running inference.

For completeness, the training scripts used in this project are included below:

```bash
python models/train_zero_shot.py
python models/train_fine_tuned.py
python models/train_combined.py
```

---

## Demo / Inference

The demo script runs inference using a **default pretrained model**
(`checkpoints/combined.pth`), which was trained on a combination of synthetic and real data
and achieved the best overall performance.

The demo supports running inference on:
- A single chessboard image
- A directory of images

Examples:

Run inference on a single image:
```bash
python demo.py \
  --input path/to/image_or_folder \
  --output_dir results/
```


---

## Reproducing Results

The training datasets (synthetic and real chessboard images) are not included in this
repository due to size constraints and course data distribution policies.

To reproduce the reported results, run the demo script using the default pretrained model
(`checkpoints/combined.pth`) on any chessboard image, as described above.

The demo internally uses the required evaluation API function
`predict_board(image: np.ndarray)` without modifying its signature or output format.
