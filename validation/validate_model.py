"""
Model Validation Script for VisionUX AI Saliency Predictor
============================================================

This script validates the AI saliency prediction model against human eye-tracking
benchmarks from established datasets like UEyes or MASSVIS.

Metrics Implemented:
- Pearson Correlation Coefficient (CC)
- Similarity Score (SIM)
- Normalized Scanpath Saliency (NSS)
- AUC-Judd (Area Under ROC Curve)

Author: Liana Baratian
Date: February 2026
"""

import os
import sys
import shutil
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for importing the model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration settings for validation pipeline."""
    
    # Input dimensions for the model
    INPUT_SIZE = (224, 224)
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    TEST_SAMPLES_DIR = DATA_DIR / "test_samples"
    RESULTS_DIR = BASE_DIR / "validation" / "results"
    
    # Dataset structure (adjust based on your dataset)
    ORIGINAL_IMAGES_SUBDIR = "stimuli"
    GROUND_TRUTH_SUBDIR = "fixation_maps"
    
    # Validation settings
    TOP_N_RESULTS = 10
    RESULT_GRID_COLS = 3  # Original | Ground Truth | Prediction


# =============================================================================
# MODEL LOADER
# =============================================================================

class VGG16FeatureExtractor(nn.Module):
    """
    VGG16-based feature extractor for pseudo-saliency map generation.
    
    Uses pre-trained VGG16 convolutional layers to extract features
    and generate attention heatmaps.
    """
    
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.extract_layers = nn.Sequential(*list(self.features.children())[:23])

    def forward(self, x: torch.Tensor) -> np.ndarray:
        """
        Generate saliency map from input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Normalized saliency map as numpy array
        """
        with torch.no_grad():
            feats = self.extract_layers(x)
            saliency = feats.mean(dim=1, keepdim=True)
            saliency = nn.functional.interpolate(
                saliency, 
                size=(x.shape[2], x.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            return saliency.squeeze().cpu().numpy()


def load_model(device: torch.device) -> VGG16FeatureExtractor:
    """Load and prepare the saliency prediction model."""
    model = VGG16FeatureExtractor()
    model.eval()
    model.to(device)
    return model


def get_transform() -> transforms.Compose:
    """Get preprocessing transforms for the model."""
    return transforms.Compose([
        transforms.Resize(Config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =============================================================================
# DATA ORGANIZATION
# =============================================================================

def organize_mit1003_dataset(
    raw_dataset_path: str,
    output_dir: Optional[Path] = None,
    max_samples: int = 100
) -> List[Dict[str, Path]]:
    """
    Organize MIT1003 dataset by pairing stimuli with fixation maps.
    
    Args:
        raw_dataset_path: Path to the MIT1003 folder
        output_dir: Output directory for organized data
        max_samples: Maximum number of samples to process
        
    Returns:
        List of dictionaries with 'original' and 'ground_truth' paths
    """
    raw_path = Path(raw_dataset_path)
    output_dir = output_dir or Config.TEST_SAMPLES_DIR
    
    # Create output directories
    originals_dir = output_dir / "originals"
    ground_truth_dir = output_dir / "ground_truth"
    originals_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    paired_data = []
    
    # MIT1003 structure: ALLSTIMULI/ and ALLFIXATIONMAPS/
    stimuli_path = raw_path / "ALLSTIMULI"
    fixation_path = raw_path / "ALLFIXATIONMAPS"
    
    if stimuli_path.exists() and fixation_path.exists():
        print(f"Found MIT1003 structure in {raw_path}")
        stimuli_files = sorted(list(stimuli_path.glob("*.jpeg")) + list(stimuli_path.glob("*.jpg")))[:max_samples]
        
        for stim_file in tqdm(stimuli_files, desc="Organizing MIT1003 dataset"):
            base_name = stim_file.stem
            
            # MIT1003 uses _fixMap suffix
            fixation_file = fixation_path / f"{base_name}_fixMap.jpg"
            
            if fixation_file.exists():
                new_orig = originals_dir / f"{base_name}.jpg"
                new_gt = ground_truth_dir / f"{base_name}.jpg"
                
                shutil.copy2(stim_file, new_orig)
                shutil.copy2(fixation_file, new_gt)
                
                paired_data.append({
                    'original': new_orig,
                    'ground_truth': new_gt,
                    'name': base_name
                })
        
        print(f"✓ Organized {len(paired_data)} MIT1003 image pairs")
    else:
        print(f"MIT1003 structure not found. Expected ALLSTIMULI/ and ALLFIXATIONMAPS/ folders.")
    
    return paired_data


def organize_dataset(
    raw_dataset_path: str,
    output_dir: Optional[Path] = None,
    stimuli_pattern: str = "*.png",
    fixation_pattern: str = "*_fixation.png"
) -> List[Dict[str, Path]]:
    """
    Organize dataset by pairing original images with ground truth fixation maps.
    
    Args:
        raw_dataset_path: Path to the raw dataset folder
        output_dir: Output directory for organized data (default: Config.TEST_SAMPLES_DIR)
        stimuli_pattern: Glob pattern for stimulus images
        fixation_pattern: Glob pattern for fixation maps
        
    Returns:
        List of dictionaries with 'original' and 'ground_truth' paths
    """
    raw_path = Path(raw_dataset_path)
    output_dir = output_dir or Config.TEST_SAMPLES_DIR
    
    # Check for MIT1003 structure first
    if (raw_path / "ALLSTIMULI").exists() and (raw_path / "ALLFIXATIONMAPS").exists():
        return organize_mit1003_dataset(raw_dataset_path, output_dir)
    
    # Create output directories
    originals_dir = output_dir / "originals"
    ground_truth_dir = output_dir / "ground_truth"
    originals_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    paired_data = []
    
    # Try to find stimuli and fixation maps
    # Adjust these patterns based on your dataset structure
    
    # Pattern 1: UEyes structure (stimuli/ and fixation_maps/ subdirs)
    stimuli_path = raw_path / Config.ORIGINAL_IMAGES_SUBDIR
    fixation_path = raw_path / Config.GROUND_TRUTH_SUBDIR
    
    if stimuli_path.exists() and fixation_path.exists():
        print(f"Found UEyes-style structure in {raw_path}")
        stimuli_files = sorted(stimuli_path.glob(stimuli_pattern))
        
        for stim_file in tqdm(stimuli_files, desc="Organizing dataset"):
            # Find matching fixation map
            base_name = stim_file.stem
            
            # Try different naming conventions
            possible_fixation_names = [
                f"{base_name}_fixation.png",
                f"{base_name}_fixmap.png",
                f"{base_name}.png",
                f"{base_name}_heatmap.png",
            ]
            
            fixation_file = None
            for fix_name in possible_fixation_names:
                candidate = fixation_path / fix_name
                if candidate.exists():
                    fixation_file = candidate
                    break
            
            if fixation_file:
                # Copy to organized directory
                new_orig = originals_dir / f"{base_name}.png"
                new_gt = ground_truth_dir / f"{base_name}.png"
                
                shutil.copy2(stim_file, new_orig)
                shutil.copy2(fixation_file, new_gt)
                
                paired_data.append({
                    'original': new_orig,
                    'ground_truth': new_gt,
                    'name': base_name
                })
    
    # Pattern 2: MASSVIS or flat structure
    else:
        print(f"Scanning {raw_path} for image pairs...")
        all_images = list(raw_path.rglob("*.png")) + list(raw_path.rglob("*.jpg"))
        
        # Group by base name
        image_groups = {}
        for img_path in all_images:
            base = img_path.stem.replace('_fixation', '').replace('_fixmap', '').replace('_heatmap', '').replace('_fixMap', '').replace('_fixPts', '')
            if base not in image_groups:
                image_groups[base] = {'stimuli': None, 'fixation': None}
            
            if any(x in img_path.stem.lower() for x in ['fixation', 'fixmap', 'heatmap', 'ground_truth']) or any(x in img_path.stem for x in ['_fixMap', '_fixPts']):
                image_groups[base]['fixation'] = img_path
            else:
                image_groups[base]['stimuli'] = img_path
        
        for base_name, files in tqdm(image_groups.items(), desc="Organizing dataset"):
            if files['stimuli'] and files['fixation']:
                new_orig = originals_dir / f"{base_name}.png"
                new_gt = ground_truth_dir / f"{base_name}.png"
                
                shutil.copy2(files['stimuli'], new_orig)
                shutil.copy2(files['fixation'], new_gt)
                
                paired_data.append({
                    'original': new_orig,
                    'ground_truth': new_gt,
                    'name': base_name
                })
    
    print(f"\n✓ Organized {len(paired_data)} image pairs in {output_dir}")
    return paired_data


def load_existing_test_samples() -> List[Dict[str, Path]]:
    """Load already organized test samples."""
    originals_dir = Config.TEST_SAMPLES_DIR / "originals"
    ground_truth_dir = Config.TEST_SAMPLES_DIR / "ground_truth"
    
    if not originals_dir.exists():
        return []
    
    paired_data = []
    # Support both PNG and JPG formats
    for orig_file in sorted(list(originals_dir.glob("*.png")) + list(originals_dir.glob("*.jpg")) + list(originals_dir.glob("*.jpeg"))):
        # Try matching extensions
        gt_file = None
        for ext in [orig_file.suffix, '.png', '.jpg', '.jpeg']:
            candidate = ground_truth_dir / f"{orig_file.stem}{ext}"
            if candidate.exists():
                gt_file = candidate
                break
        
        if gt_file:
            paired_data.append({
                'original': orig_file,
                'ground_truth': gt_file,
                'name': orig_file.stem
            })
    
    return paired_data


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_image(
    image_path: Path, 
    target_size: Tuple[int, int] = Config.INPUT_SIZE,
    maintain_aspect: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image for model input while maintaining spatial integrity.
    
    Args:
        image_path: Path to the image file
        target_size: Target dimensions (height, width)
        maintain_aspect: Whether to maintain aspect ratio with padding
        
    Returns:
        Tuple of (preprocessed_image, original_image)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original = img.copy()
    
    if maintain_aspect:
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        preprocessed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        preprocessed = cv2.resize(img, (target_size[1], target_size[0]))
    
    return preprocessed, original


def preprocess_saliency_map(
    saliency_path: Path,
    target_size: Tuple[int, int] = Config.INPUT_SIZE
) -> np.ndarray:
    """
    Preprocess ground truth saliency/fixation map.
    
    Args:
        saliency_path: Path to the saliency map
        target_size: Target dimensions
        
    Returns:
        Normalized saliency map
    """
    saliency = cv2.imread(str(saliency_path), cv2.IMREAD_GRAYSCALE)
    if saliency is None:
        raise ValueError(f"Could not load saliency map: {saliency_path}")
    
    # Resize to target size
    saliency = cv2.resize(saliency, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1]
    saliency = saliency.astype(np.float32)
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_cc(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Pearson Correlation Coefficient (CC).
    
    Higher is better. Range: [-1, 1]
    
    Args:
        pred: Predicted saliency map
        gt: Ground truth saliency map
        
    Returns:
        Correlation coefficient
    """
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Remove mean
    pred_centered = pred_flat - pred_flat.mean()
    gt_centered = gt_flat - gt_flat.mean()
    
    # Calculate correlation
    numerator = np.sum(pred_centered * gt_centered)
    denominator = np.sqrt(np.sum(pred_centered**2) * np.sum(gt_centered**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def calculate_sim(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Similarity Score (SIM).
    
    Measures histogram intersection between normalized maps.
    Higher is better. Range: [0, 1]
    
    Args:
        pred: Predicted saliency map
        gt: Ground truth saliency map
        
    Returns:
        Similarity score
    """
    # Normalize to probability distributions
    pred_norm = pred / (pred.sum() + 1e-8)
    gt_norm = gt / (gt.sum() + 1e-8)
    
    # Calculate histogram intersection (element-wise minimum, then sum)
    return np.sum(np.minimum(pred_norm, gt_norm))


def calculate_nss(pred: np.ndarray, fixation_map: np.ndarray) -> float:
    """
    Calculate Normalized Scanpath Saliency (NSS).
    
    Measures saliency values at fixation locations.
    Higher is better.
    
    Args:
        pred: Predicted saliency map
        fixation_map: Binary fixation map (or continuous, will be thresholded)
        
    Returns:
        NSS score
    """
    # Normalize prediction to zero mean and unit std
    pred_normalized = (pred - pred.mean()) / (pred.std() + 1e-8)
    
    # Threshold fixation map to get binary fixations
    fixation_binary = (fixation_map > fixation_map.mean()).astype(np.float32)
    
    if fixation_binary.sum() == 0:
        return 0.0
    
    # Calculate mean saliency at fixation points
    return float(np.sum(pred_normalized * fixation_binary) / fixation_binary.sum())


def calculate_auc_judd(pred: np.ndarray, fixation_map: np.ndarray) -> float:
    """
    Calculate AUC-Judd (Area Under ROC Curve with Judd's method).
    
    Higher is better. Range: [0, 1], 0.5 = chance
    
    Args:
        pred: Predicted saliency map
        fixation_map: Ground truth fixation map
        
    Returns:
        AUC score
    """
    # Get fixation points
    fixation_binary = fixation_map > fixation_map.mean()
    fixation_points = pred[fixation_binary]
    non_fixation_points = pred[~fixation_binary]
    
    if len(fixation_points) == 0 or len(non_fixation_points) == 0:
        return 0.5
    
    # Calculate AUC
    n_fix = len(fixation_points)
    n_non = len(non_fixation_points)
    
    # For each fixation point, count how many non-fixation points have lower saliency
    auc = 0.0
    for fix_val in fixation_points:
        auc += np.sum(non_fixation_points < fix_val)
        auc += 0.5 * np.sum(non_fixation_points == fix_val)
    
    auc /= (n_fix * n_non)
    return float(auc)


def calculate_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Calculate all saliency metrics.
    
    Args:
        pred: Predicted saliency map
        gt: Ground truth saliency map
        
    Returns:
        Dictionary with all metric scores
    """
    return {
        'CC': calculate_cc(pred, gt),
        'SIM': calculate_sim(pred, gt),
        'NSS': calculate_nss(pred, gt),
        'AUC': calculate_auc_judd(pred, gt)
    }


# =============================================================================
# PREDICTION
# =============================================================================

def predict_saliency(
    model: VGG16FeatureExtractor,
    image: np.ndarray,
    device: torch.device,
    transform: transforms.Compose
) -> np.ndarray:
    """
    Generate saliency prediction for an image.
    
    Args:
        model: The saliency prediction model
        image: Input image (BGR format from OpenCV)
        device: Torch device
        transform: Preprocessing transforms
        
    Returns:
        Predicted saliency map
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Transform and predict
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    saliency = model(input_tensor)
    
    return saliency


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_result_grid(
    results: List[Dict],
    output_path: Path,
    top_n: int = Config.TOP_N_RESULTS
) -> None:
    """
    Generate a result grid image showing comparisons.
    
    Args:
        results: List of result dictionaries with images and metrics
        output_path: Path to save the grid image
        top_n: Number of top results to include
    """
    # Sort by CC score (best first)
    sorted_results = sorted(results, key=lambda x: x['metrics']['CC'], reverse=True)[:top_n]
    
    n_rows = len(sorted_results)
    n_cols = 3  # Original | Ground Truth | Prediction
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    column_titles = ['Original Image', 'Human Ground Truth', 'AI Prediction']
    
    for row_idx, result in enumerate(sorted_results):
        # Original image
        orig = cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB)
        axes[row_idx, 0].imshow(orig)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(column_titles[0], fontsize=14, fontweight='bold')
        
        # Ground truth
        axes[row_idx, 1].imshow(result['ground_truth'], cmap='jet')
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title(column_titles[1], fontsize=14, fontweight='bold')
        
        # Prediction
        axes[row_idx, 2].imshow(result['prediction'], cmap='jet')
        axes[row_idx, 2].axis('off')
        if row_idx == 0:
            axes[row_idx, 2].set_title(column_titles[2], fontsize=14, fontweight='bold')
        
        # Add metrics label
        metrics = result['metrics']
        metrics_text = f"CC: {metrics['CC']:.3f} | SIM: {metrics['SIM']:.3f} | NSS: {metrics['NSS']:.2f}"
        axes[row_idx, 2].text(
            0.5, -0.1, metrics_text,
            transform=axes[row_idx, 2].transAxes,
            ha='center', fontsize=10, color='gray'
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Result grid saved to {output_path}")


def create_metrics_summary(
    all_metrics: List[Dict[str, float]],
    output_path: Path
) -> Dict[str, Dict[str, float]]:
    """
    Create a summary of all metrics across the dataset.
    
    Args:
        all_metrics: List of metric dictionaries
        output_path: Path to save the summary
        
    Returns:
        Summary statistics dictionary
    """
    summary = {}
    metric_names = ['CC', 'SIM', 'NSS', 'AUC']
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    # Save summary to file
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VisionUX AI - Model Validation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset Size: {len(all_metrics)} samples\n\n")
        
        f.write("Metric Results (Mean ± Std):\n")
        f.write("-" * 40 + "\n")
        
        for metric, stats in summary.items():
            f.write(f"  {metric:6s}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"          (min: {stats['min']:.4f}, max: {stats['max']:.4f})\n\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Metric Interpretations:\n")
        f.write("-" * 40 + "\n")
        f.write("  CC  (Correlation): Higher is better, range [-1, 1]\n")
        f.write("  SIM (Similarity):  Higher is better, range [0, 1]\n")
        f.write("  NSS (Scanpath):    Higher is better, typically [0, 3+]\n")
        f.write("  AUC (ROC Area):    Higher is better, 0.5 = chance\n")
        f.write("=" * 60 + "\n")
    
    print(f"✓ Metrics summary saved to {output_path}")
    return summary


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def run_validation(
    dataset_path: Optional[str] = None,
    use_existing: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run the complete validation pipeline.
    
    Args:
        dataset_path: Path to raw dataset (optional if using existing)
        use_existing: Whether to use already organized test samples
        
    Returns:
        Summary of validation metrics
    """
    print("\n" + "=" * 60)
    print("VisionUX AI - Model Validation Pipeline")
    print("=" * 60 + "\n")
    
    # Setup
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\n[1/5] Loading model...")
    model = load_model(device)
    transform = get_transform()
    print("✓ Model loaded successfully")
    
    # Get test data
    print("\n[2/5] Loading test data...")
    if use_existing:
        paired_data = load_existing_test_samples()
    
    if not paired_data and dataset_path:
        paired_data = organize_dataset(dataset_path)
    
    if not paired_data:
        print("✗ No test data found. Please provide a dataset path or organize data first.")
        print("\nUsage:")
        print("  1. Place your dataset in a folder with 'stimuli' and 'fixation_maps' subdirs")
        print("  2. Run: run_validation(dataset_path='/path/to/your/dataset')")
        return {}
    
    print(f"✓ Found {len(paired_data)} test samples")
    
    # Run predictions and calculate metrics
    print("\n[3/5] Running predictions and calculating metrics...")
    all_results = []
    all_metrics = []
    
    for item in tqdm(paired_data, desc="Processing"):
        try:
            # Load and preprocess
            preprocessed, original = preprocess_image(item['original'])
            ground_truth = preprocess_saliency_map(item['ground_truth'])
            
            # Predict
            prediction = predict_saliency(model, preprocessed, device, transform)
            
            # Resize prediction to match ground truth if needed
            if prediction.shape != ground_truth.shape:
                prediction = cv2.resize(prediction, (ground_truth.shape[1], ground_truth.shape[0]))
            
            # Calculate metrics
            metrics = calculate_all_metrics(prediction, ground_truth)
            all_metrics.append(metrics)
            
            # Store results for visualization
            all_results.append({
                'name': item['name'],
                'original': preprocessed,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"Warning: Failed to process {item['name']}: {e}")
            continue
    
    if not all_metrics:
        print("✗ No samples were successfully processed")
        return {}
    
    # Generate result grid
    print("\n[4/5] Generating result grid...")
    grid_path = Config.RESULTS_DIR / "validation_result_grid.png"
    create_result_grid(all_results, grid_path)
    
    # Generate metrics summary
    print("\n[5/5] Generating metrics summary...")
    summary_path = Config.RESULTS_DIR / "validation_metrics_summary.txt"
    summary = create_metrics_summary(all_metrics, summary_path)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nSamples Processed: {len(all_metrics)}")
    print("\nMetric Results (Mean ± Std):")
    print("-" * 40)
    for metric, stats in summary.items():
        print(f"  {metric:6s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("\n" + "=" * 60)
    
    return summary


# =============================================================================
# DEMO WITH SYNTHETIC DATA
# =============================================================================

def create_demo_data() -> None:
    """Create synthetic demo data for testing the validation pipeline."""
    print("\n[DEMO] Creating synthetic test data...")
    
    demo_dir = Config.TEST_SAMPLES_DIR
    originals_dir = demo_dir / "originals"
    ground_truth_dir = demo_dir / "ground_truth"
    
    originals_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 synthetic samples
    for i in range(5):
        # Create a random "UI" image
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Add some shapes to simulate UI elements
        cv2.rectangle(img, (20, 20), (100, 60), (255, 100, 100), -1)
        cv2.rectangle(img, (120, 80), (200, 120), (100, 255, 100), -1)
        cv2.circle(img, (112, 180), 30, (100, 100, 255), -1)
        
        # Create synthetic fixation map (centered attention)
        fixation = np.zeros((224, 224), dtype=np.float32)
        y, x = np.ogrid[:224, :224]
        center_y, center_x = 112, 112
        fixation = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 50**2))
        
        # Add some random fixation points
        for _ in range(3):
            py, px = np.random.randint(30, 194, 2)
            fixation += 0.5 * np.exp(-((x - px)**2 + (y - py)**2) / (2 * 30**2))
        
        fixation = (fixation / fixation.max() * 255).astype(np.uint8)
        
        # Save
        cv2.imwrite(str(originals_dir / f"demo_{i+1}.png"), img)
        cv2.imwrite(str(ground_truth_dir / f"demo_{i+1}.png"), fixation)
    
    print(f"✓ Created 5 demo samples in {demo_dir}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate VisionUX AI saliency model against eye-tracking benchmarks"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Path to the raw dataset folder"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic demo data"
    )
    parser.add_argument(
        "--organize-only",
        action="store_true",
        help="Only organize the dataset, don't run validation"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        create_demo_data()
        run_validation(use_existing=True)
    elif args.organize_only and args.dataset:
        organize_dataset(args.dataset)
    else:
        run_validation(dataset_path=args.dataset, use_existing=True)
