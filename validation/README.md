# VisionUX AI - Model Validation Module

## Scientific Validation Against Human Eye-Tracking Benchmarks

This module validates the VisionUX AI saliency prediction model against established human eye-tracking datasets such as **UEyes** and **MASSVIS**.

---

## 📊 Implemented Metrics

| Metric                                 | Description                                            | Range    | Interpretation   |
| -------------------------------------- | ------------------------------------------------------ | -------- | ---------------- |
| **CC** (Correlation Coefficient)       | Pearson correlation between predicted and ground truth | [-1, 1]  | Higher is better |
| **SIM** (Similarity Score)             | Histogram intersection of normalized maps              | [0, 1]   | Higher is better |
| **NSS** (Normalized Scanpath Saliency) | Saliency values at human fixation points               | [0, 3+]  | Higher is better |
| **AUC** (Area Under ROC Curve)         | Classification accuracy of fixation prediction         | [0.5, 1] | 0.5 = chance     |

---

## 🚀 Quick Start

### 1. Run with Demo Data

```bash
cd validation
python validate_model.py --demo
```

### 2. Run with UEyes/MASSVIS Dataset

```bash
python validate_model.py --dataset /path/to/ueyes_dataset
```

### 3. Organize Dataset Only

```bash
python validate_model.py --dataset /path/to/dataset --organize-only
```

---

## 📁 Expected Dataset Structure

### UEyes Format

```
ueyes_dataset/
├── stimuli/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── fixation_maps/
    ├── image_001_fixation.png
    ├── image_002_fixation.png
    └── ...
```

### MASSVIS Format

```
massvis_dataset/
├── image_001.png
├── image_001_fixmap.png
├── image_002.png
├── image_002_fixmap.png
└── ...
```

---

## 📈 Output

After running validation, you'll find:

### 1. Result Grid (`results/validation_result_grid.png`)

Visual comparison showing:

- **Column 1**: Original UI Screenshot
- **Column 2**: Human Ground Truth (Eye-Tracking Data)
- **Column 3**: AI Prediction

### 2. Metrics Summary (`results/validation_metrics_summary.txt`)

```
============================================================
VisionUX AI - Model Validation Summary
============================================================

Dataset Size: 100 samples

Metric Results (Mean ± Std):
----------------------------------------
  CC    : 0.7234 ± 0.0891
  SIM   : 0.6512 ± 0.0734
  NSS   : 1.8923 ± 0.4521
  AUC   : 0.8156 ± 0.0623
============================================================
```

---

## 🔬 Research References

This validation methodology follows established saliency benchmarking practices from:

1. **UEyes Dataset**: Leiva et al., "Understanding Visual Attention in Mobile User Interfaces"
2. **MASSVIS**: Borkin et al., "What Makes a Visualization Memorable?"
3. **MIT Saliency Benchmark**: Bylinskii et al., "MIT/Tuebingen Saliency Benchmark"

---

## 📋 Usage in Portfolio

When presenting to recruiters, emphasize:

✅ **Scientific Validation**: Model tested against real human eye-tracking data  
✅ **Industry Metrics**: Using standard metrics (CC, SIM, NSS, AUC) from peer-reviewed research  
✅ **Benchmark Datasets**: Validated on UEyes/MASSVIS, used by Google, MIT, and Adobe  
✅ **Quantifiable Results**: Concrete numbers demonstrating model performance

---

## 🛠️ API Reference

### Python API

```python
from validation.validate_model import (
    run_validation,
    organize_dataset,
    calculate_all_metrics
)

# Run full validation pipeline
summary = run_validation(dataset_path="/path/to/dataset")

# Just organize a dataset
pairs = organize_dataset("/path/to/raw_dataset")

# Calculate metrics for single prediction
metrics = calculate_all_metrics(prediction, ground_truth)
print(f"Correlation: {metrics['CC']:.4f}")
print(f"Similarity: {metrics['SIM']:.4f}")
```

---

## 📦 Dependencies

- `numpy`
- `scipy`
- `opencv-python`
- `matplotlib`
- `torch`
- `torchvision`
- `tqdm`
- `Pillow`

Install with:

```bash
pip install numpy scipy opencv-python matplotlib torch torchvision tqdm Pillow
```
