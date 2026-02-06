"""
Training script for saliency prediction model using SALICON dataset.
SALICON: 10,000 training + 5,000 validation images from MS COCO with mouse-tracking fixations.
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# --- Dataset ---
class SALICONDataset(Dataset):
    """SALICON Saliency Dataset"""
    
    def __init__(self, images_dir, annotations_file, transform=None, target_size=224):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Load annotations
        print(f"Loading annotations from {annotations_file}...")
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # Build image lookup
        self.images = {img['id']: img for img in data['images']}
        
        # Group fixations by image
        self.fixations_by_image = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.fixations_by_image:
                self.fixations_by_image[img_id] = []
            self.fixations_by_image[img_id].extend(ann['fixations'])
        
        # Filter to images that exist
        self.valid_ids = []
        for img_id in self.fixations_by_image.keys():
            img_info = self.images.get(img_id)
            if img_info:
                img_path = self.images_dir / img_info['file_name']
                if img_path.exists():
                    self.valid_ids.append(img_id)
        
        print(f"Found {len(self.valid_ids)} valid images with fixations")
    
    def __len__(self):
        return len(self.valid_ids)
    
    def _create_fixation_map(self, fixations, height, width):
        """Create a fixation density map from fixation points"""
        fix_map = np.zeros((height, width), dtype=np.float32)
        
        for fix in fixations:
            row, col = fix[0], fix[1]
            # Clamp to image bounds (1-indexed to 0-indexed)
            row = max(0, min(height - 1, int(row) - 1))
            col = max(0, min(width - 1, int(col) - 1))
            fix_map[row, col] += 1
        
        # Apply Gaussian blur to create smooth saliency map
        if fix_map.max() > 0:
            # Kernel size based on image dimensions (~1 degree visual angle)
            kernel_size = max(25, int(min(height, width) * 0.05))
            if kernel_size % 2 == 0:
                kernel_size += 1
            fix_map = cv2.GaussianBlur(fix_map, (kernel_size, kernel_size), 0)
            # Normalize to [0, 1]
            fix_map = fix_map / (fix_map.max() + 1e-8)
        
        return fix_map
    
    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.images_dir / img_info['file_name']
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        
        # Create fixation map at original resolution
        fixations = self.fixations_by_image[img_id]
        fix_map = self._create_fixation_map(fixations, orig_height, orig_width)
        
        # Resize fixation map to target size
        fix_map = cv2.resize(fix_map, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        fix_map = torch.from_numpy(fix_map).unsqueeze(0).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, fix_map


# --- Model ---
class SaliencyNet(nn.Module):
    """
    VGG16-based encoder-decoder network for saliency prediction.
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Encoder: VGG16 features
        vgg = models.vgg16(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:23])
        
        # Decoder: Upsample to original resolution
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        saliency = self.decoder(features)
        return saliency


# --- Loss Functions ---
class SaliencyLoss(nn.Module):
    """Combined loss for saliency prediction"""
    
    def __init__(self, kl_weight=1.0, bce_weight=1.0, cc_weight=0.5):
        super().__init__()
        self.kl_weight = kl_weight
        self.bce_weight = bce_weight
        self.cc_weight = cc_weight
        self.bce = nn.BCELoss()
    
    def kl_divergence(self, pred, target):
        pred = pred / (pred.sum(dim=(2, 3), keepdim=True) + 1e-8)
        target = target / (target.sum(dim=(2, 3), keepdim=True) + 1e-8)
        kl = target * torch.log(target / (pred + 1e-8) + 1e-8)
        return kl.sum(dim=(2, 3)).mean()
    
    def correlation_coefficient(self, pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1) + 1e-8)
        
        cc = numerator / denominator
        return -cc.mean()
    
    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = nn.functional.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        bce_loss = self.bce(pred, target)
        kl_loss = self.kl_divergence(pred, target)
        cc_loss = self.correlation_coefficient(pred, target)
        
        total_loss = self.bce_weight * bce_loss + self.kl_weight * kl_loss + self.cc_weight * cc_loss
        return total_loss, bce_loss, kl_loss, cc_loss


# --- Training ---
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    total_bce = 0
    total_kl = 0
    total_cc = 0
    num_batches = len(dataloader)
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{total_epochs} - TRAINING")
    print(f"{'='*60}")
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        step = batch_idx + 1
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss, bce, kl, cc = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_bce += bce.item()
        total_kl += kl.item()
        total_cc += cc.item()
        
        if step % 50 == 0 or step == num_batches:
            progress = (step / num_batches) * 100
            print(f"  Step [{step}/{num_batches}] ({progress:.0f}%) | Loss: {loss.item():.4f} | BCE: {bce.item():.4f} | CC: {-cc.item():.4f}")
    
    n = len(dataloader)
    return total_loss / n, total_bce / n, total_kl / n, -total_cc / n


def validate(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    total_cc = 0
    num_batches = len(dataloader)
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{total_epochs} - VALIDATION")
    print(f"{'='*60}")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            step = batch_idx + 1
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss, _, _, cc = criterion(outputs, targets)
            
            total_loss += loss.item()
            total_cc += cc.item()
            
            if step % 20 == 0 or step == num_batches:
                progress = (step / num_batches) * 100
                print(f"  Step [{step}/{num_batches}] ({progress:.0f}%) | Val Loss: {loss.item():.4f}")
    
    n = len(dataloader)
    return total_loss / n, -total_cc / n


def main():
    parser = argparse.ArgumentParser(description="Train saliency model on SALICON")
    parser.add_argument("--data_dir", type=str, default="data/SALICON", help="Path to SALICON dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_images = os.path.join(args.data_dir, "train2014")
    train_ann = os.path.join(args.data_dir, "fixations_train2014.json")
    val_images = os.path.join(args.data_dir, "val2014")
    val_ann = os.path.join(args.data_dir, "fixations_val2014.json")
    
    train_dataset = SALICONDataset(
        images_dir=train_images,
        annotations_file=train_ann,
        transform=img_transform,
        target_size=args.img_size
    )
    
    val_dataset = SALICONDataset(
        images_dir=val_images,
        annotations_file=val_ann,
        transform=img_transform,
        target_size=args.img_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model
    model = SaliencyNet(pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = SaliencyLoss(kl_weight=1.0, bce_weight=1.0, cc_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("STARTING TRAINING ON SALICON")
    print(f"Total Epochs: {args.epochs}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("="*60)
    
    for epoch in range(args.epochs):
        current_epoch = epoch + 1
        
        train_loss, train_bce, train_kl, train_cc = train_epoch(
            model, train_loader, criterion, optimizer, device, current_epoch, args.epochs
        )
        val_loss, val_cc = validate(model, val_loader, criterion, device, current_epoch, args.epochs)
        
        scheduler.step(val_loss)
        
        print(f"\n{'*'*60}")
        print(f"EPOCH {current_epoch}/{args.epochs} SUMMARY")
        print(f"{'*'*60}")
        print(f"  Train Loss: {train_loss:.4f} | BCE: {train_bce:.4f} | KL: {train_kl:.4f} | CC: {train_cc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val CC: {val_cc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cc': val_cc,
            }, os.path.join(args.save_dir, "best_saliency_model.pth"))
            print(f"\n  ✓ NEW BEST MODEL SAVED! (val_loss: {val_loss:.4f})")
        
        if current_epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, f"checkpoint_epoch_{current_epoch}.pth"))
            print(f"  ✓ Checkpoint saved at epoch {current_epoch}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(args.save_dir, 'best_saliency_model.pth')}")
    print("="*60)


if __name__ == "__main__":
    main()
