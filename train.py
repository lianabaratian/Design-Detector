"""
Training script for saliency prediction model using MIT1003 dataset.
This script fine-tunes a VGG16-based encoder-decoder network for saliency prediction.
"""

import os
import glob
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
from tqdm import tqdm


# --- Dataset ---
class MIT1003Dataset(Dataset):
    """MIT1003 Saliency Dataset"""
    
    def __init__(self, stimuli_dir, fixation_dir, transform=None, target_transform=None):
        self.stimuli_dir = Path(stimuli_dir)
        self.fixation_dir = Path(fixation_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Get all stimulus images
        self.stimuli = sorted(glob.glob(str(self.stimuli_dir / "*.jpeg")))
        
        # Build mapping to fixation maps
        self.pairs = []
        for stim_path in self.stimuli:
            stim_name = Path(stim_path).stem
            # Fixation map naming convention: {name}_fixMap.jpg
            fix_map_path = self.fixation_dir / f"{stim_name}_fixMap.jpg"
            if fix_map_path.exists():
                self.pairs.append((stim_path, str(fix_map_path)))
        
        print(f"Found {len(self.pairs)} image-fixation pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        stim_path, fix_path = self.pairs[idx]
        
        # Load images
        image = Image.open(stim_path).convert("RGB")
        fixation = Image.open(fix_path).convert("L")  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            fixation = self.target_transform(fixation)
        
        # Normalize fixation map to [0, 1]
        fixation = fixation / 255.0 if fixation.max() > 1 else fixation
        
        return image, fixation


# --- Model ---
class SaliencyNet(nn.Module):
    """
    VGG16-based encoder-decoder network for saliency prediction.
    Encoder: VGG16 pretrained features
    Decoder: Upsampling layers to produce saliency map
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
    """Combined loss for saliency prediction: KL divergence + BCE + CC"""
    
    def __init__(self, kl_weight=1.0, bce_weight=1.0, cc_weight=0.5):
        super().__init__()
        self.kl_weight = kl_weight
        self.bce_weight = bce_weight
        self.cc_weight = cc_weight
        self.bce = nn.BCELoss()
    
    def kl_divergence(self, pred, target):
        """KL divergence between predicted and target saliency maps"""
        pred = pred / (pred.sum(dim=(2, 3), keepdim=True) + 1e-8)
        target = target / (target.sum(dim=(2, 3), keepdim=True) + 1e-8)
        kl = target * torch.log(target / (pred + 1e-8) + 1e-8)
        return kl.sum(dim=(2, 3)).mean()
    
    def correlation_coefficient(self, pred, target):
        """Pearson correlation coefficient (higher is better, so we negate for loss)"""
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1) + 1e-8)
        
        cc = numerator / denominator
        return -cc.mean()  # Negative because we want to maximize CC
    
    def forward(self, pred, target):
        # Resize prediction to match target if needed
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
        
        # Print progress every 10 batches or at the end
        if step % 10 == 0 or step == num_batches:
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
            
            if step % 5 == 0 or step == num_batches:
                progress = (step / num_batches) * 100
                print(f"  Step [{step}/{num_batches}] ({progress:.0f}%) | Val Loss: {loss.item():.4f}")
    
    n = len(dataloader)
    return total_loss / n, -total_cc / n


def main():
    parser = argparse.ArgumentParser(description="Train saliency model on MIT1003")
    parser.add_argument("--data_dir", type=str, default="data/MIT1003", help="Path to MIT1003 dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # Dataset
    stimuli_dir = os.path.join(args.data_dir, "ALLSTIMULI")
    fixation_dir = os.path.join(args.data_dir, "ALLFIXATIONMAPS")
    
    full_dataset = MIT1003Dataset(
        stimuli_dir=stimuli_dir,
        fixation_dir=fixation_dir,
        transform=img_transform,
        target_transform=target_transform
    )
    
    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = SaliencyNet(pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = SaliencyLoss(kl_weight=1.0, bce_weight=1.0, cc_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
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
        
        # Save best model
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
        
        # Save checkpoint every 10 epochs
        if (current_epoch) % 10 == 0:
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
