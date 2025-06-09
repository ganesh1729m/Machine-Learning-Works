# Tree Crown Detection

Automated detection of tree crowns in high-resolution RGB imagery using PyTorch and Faster R-CNN.

## 📂 Repository Structure

```
tree-crown-detection/
├── LICENSE
├── README.md
├── .gitignore
├── requirements.txt
├── download_data.sh        # Kaggle download script
├── data/                  # Downloaded NEON dataset (small sample or link)
├── notebooks/
|   ├── 01-exploratory-data-analysis-tree-crown-detection.ipynb          # Jupyter notebooks (EDA, Training, Evaluation)
|   ├── 02-training-evaluation-tree-crown-dataset.ipynb
├── src/                   # Main source code modules
│   ├── dataset.py
│   ├── data_loaders.py
│   ├── model_utils.py
│   ├── train.py
│   ├── evaluate_crops.py
│   ├── evaluate_for_full_images.py
│   └── infer_full_image.py
├── configs/               # YAML configuration files
│   └── default.yaml
├── checkpoints/           # Saved model checkpoints
|   └── fasterrcnn_final_epoch30.pth
├── figures/               # Output visualizations
└── docs/                  # Report PDF
|   └── Tree_Crown_Detection_Report.pdf
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create & activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
bash download_data.sh
```

Data will be placed in `data/tree-crown-dataset/neon-dataset/`.

### 3. Configuration

Edit paths and hyperparameters in `configs/default.yaml` as needed.

### 4. Exploratory Data Analysis

Launch the EDA notebook:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 5. Training

Train the model on cropped images:

```bash
python src/train.py
```

Checkpoints will be saved to `checkpoints/`.

### 6. Evaluation

#### Crops

```bash
python src/evaluate.py src/data_loaders.py
```

#### Full Images

```bash
python src/evaluate.py /path/to/test/images /path/to/annotations checkpoints/fasterrcnn_epochXX.pth
```

### 7. Inference

Visualize predictions on a full image:

```bash
python src/infer_full_image.py /path/to/image.tif data/tree-crown-dataset/neon-dataset/annotations checkpoints/fasterrcnn_epochXX.pth
```

## 📖 Report

Full project report is in `docs/report.pdf` (LaTeX source: `docs/report.tex`).

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## ✉️ Contact

Manpurwar Ganesh — [ai22btech11017@iith.ac.in](mailto:ai22btech11017@iith.ac.in)
