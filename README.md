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
│   └── predict.py
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
git clone https://github.com/omrib1101/chess-sim2real.git
cd chess-sim2real
```

### 2. Create and activate a virtual environment

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```
**Note:**  
Synthetic data generation scripts rely on Blender's internal Python environment
(`bpy`, `mathutils`) and are not installed via pip.
These dependencies are **not required** for training or inference.

---

## Training

The training pipeline is structured into three sequential stages to bridge the sim-to-real gap.
> 💡 **Dataset Prerequisite:** Before starting the training process, ensure you have downloaded the dataset from the provided Google Drive link and extracted it locally. Please refer to the [Dataset Setup](#dataset-setup) section for download links and instructions on how to organize the folders.
Each script is designed to be standalone: if the required base model is missing from your local `/checkpoints` folder, the script will **automatically download** our official version from GitHub Releases.



### 1. Zero-Shot Training
Trains the initial model architecture using only synthetic data generated from Blender.
* **Purpose:** Establishes a baseline for generalization without seeing any real-world images.
* **Command:**
    ```bash
    python models/train_zero_shot.py --data_dir path/to/synthetic_data --output_model my_zero_shot.pth
    ```

### 2. Fine-Tuning
Fine-tunes the **Zero-Shot** model using a small set of real chessboard images.
* **Purpose:** Adapts synthetic features to real-world textures, lighting, and camera noise.
* **Logic:** Automatically attempts to load `checkpoints/zero_shot.pth` as the base.
* **Command:**
    ```bash
    python models/train_fine_tuned.py --data_dir path/to/real_data --output_model my_fine_tuned.pth
    ```

### 3. Combined Training
The final stage where the **Fine-Tuned** model is trained on a joint dataset of synthetic and real images.
* **Purpose:** Maximizes accuracy by combining the volume of synthetic data with the precision of real-world samples.
* **Logic:** Automatically attempts to load `checkpoints/fine_tuned.pth` as the base.
* **Command:**
    ```bash
    python models/train_combined.py --data_dir path/to/combined_data --output_model my_combined.pth
    ```

---

### ⚠️ Managing Checkpoints & The `--init_model` Argument

To prevent confusion and ensure the integrity of the evaluation, please note the following:

* **Protect the `/checkpoints` folder:** This directory is reserved for our "official" project weights (`zero_shot.pth`, `fine_tuned.pth`, `combined.pth`).
* **Why use `--init_model`?** While our scripts automatically download official weights if the folder is empty, you should use the `--init_model` argument when you want to **chain your own training stages**. 
    * *Example:* If you train a custom Zero-Shot model and save it as `my_zero_shot.pth`, you must pass `--init_model my_zero_shot.pth` when running the Fine-Tuning script to ensure you are building upon **your** specific results rather than our official ones.
* **Avoid Overwriting:** We strongly recommend **not** saving your generated models back into the `/checkpoints` folder using the official names. If you overwrite the files in `/checkpoints`, the `demo.py` script will load **your** experimental weights instead of our verified benchmarks.


---

### Training Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_dir` | **Required** | Path to the dataset root. Must contain `train/` and `validation/` subfolders. |
| `--init_model` | `None` | Path to a local `.pth` file to start from. If omitted, the script downloads the official base model from GitHub. |
| `--output_model` | *(Stage dependent)* | The filename for your newly trained weights. |
| `--epochs` | `20` | Number of training iterations. |

## Evaluation Function (Instructor Requirements)

As required for the course evaluation, the project implements a function
`predict_board(image: np.ndarray) -> torch.Tensor`.

### Function Behavior

- **Input**:  
  A single RGB image of a chessboard as a NumPy array of shape `(H, W, 3)`.


- **Output**:  
  A `torch.Tensor` of shape `(8, 8)` representing the predicted board state,
  following the class encoding defined in the assignment instructions.

 - **Additional Output (Visualization)**:  
  In addition to returning the board-state tensor, the function also generates
  a visual representation of the predicted chessboard and saves it.

  The output image is saved under:
  inference/results/


### Function Location

The function is implemented in:
inference/predict.py


### Example Usage

```python
import numpy as np
from inference.predict import predict_board

image = np.array(...)  # RGB image of a chessboard
board_tensor = predict_board(image)
```

After calling the function, the predicted board images are saved under `inference/results/` with unique,
automatically generated filenames.

---

## Demo

The demo script runs inference using a **default pretrained model**
(`checkpoints/combined.pth`), which was trained on a combination of synthetic and real data
and achieved the best overall performance.

The demo supports running inference on:
- A single chessboard image
- A directory of images

Note: Initial run includes an automatic download of the pre-trained model weights (~44MB) from GitHub Releases. 
The process may take a few seconds, but subsequent runs will use the locally cached file in the /checkpoints directory.

Run inference on a single image/directory of images, and save the result/s in an output directory `inference/results/`:
```bash
python demo.py --input path_to_image_or_folder
```


---

## Reproducing Results

The training datasets (synthetic and real chessboard images) are not included in this
repository due to size constraints and course data distribution policies.

To reproduce the reported results, run the demo script using the default pretrained model
(`checkpoints/combined.pth`) on any chessboard image, as described above.

The demo internally uses the required evaluation API function
`predict_board(image: np.ndarray)` without modifying its signature or output format.
