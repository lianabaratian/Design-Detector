
import io
import os
import base64
from typing import Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

app = FastAPI()

# Allow all CORS origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Saliency Model (trained on MIT1003) ---
class SaliencyNet(nn.Module):
    """
    VGG16-based encoder-decoder network for saliency prediction.
    Trained on MIT1003 eye-tracking dataset.
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


# --- Fallback Feature Extractor (if no trained model available) ---
class VGG16FeatureExtractor(nn.Module):
    """Fallback: Uses VGG16 features as pseudo-saliency (no training required)"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.extract_layers = nn.Sequential(*list(self.features.children())[:23])

    def forward(self, x):
        with torch.no_grad():
            feats = self.extract_layers(x)
            saliency = feats.mean(dim=1, keepdim=True)
            saliency = nn.functional.interpolate(saliency, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            return saliency


# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_saliency_model.pth")
USE_TRAINED_MODEL = os.path.exists(MODEL_PATH)

if USE_TRAINED_MODEL:
    print(f"Loading trained saliency model from {MODEL_PATH}")
    model = SaliencyNet(pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (val_cc: {checkpoint.get('val_cc', 'N/A'):.4f})")
else:
    print("No trained model found. Using VGG16 feature extractor as fallback.")
    print("Run 'python train.py' to train the model on MIT1003 dataset.")
    model = VGG16FeatureExtractor()

model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Utility Functions ---
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image

def resize_with_aspect_ratio(image: np.ndarray, size: int = 224) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Pad to square
    delta_w = size - new_w
    delta_h = size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale

def apply_colormap_and_blend(orig: np.ndarray, saliency: np.ndarray) -> Tuple[np.ndarray, float]:
    # Gaussian blur
    saliency_blur = cv2.GaussianBlur(saliency, (11, 11), 0)
    # Normalize to 0-255
    saliency_uint8 = np.uint8(255 * saliency_blur)
    # Apply Jet colormap
    heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    # Blend
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
    # Attention score: % of pixels above 70% intensity in saliency
    attention_score = float((saliency > 0.7).sum()) / saliency.size * 100
    return overlay, attention_score

def image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# --- API Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    pil_img = read_imagefile(contents)
    orig_np = np.array(pil_img)
    # Resize with aspect ratio and pad
    img_resized, scale = resize_with_aspect_ratio(orig_np, 224)
    # Prepare for model
    input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(device)
    
    # Generate saliency map
    with torch.no_grad():
        saliency_output = model(input_tensor)
        if USE_TRAINED_MODEL:
            # Trained model outputs [B, 1, H, W] tensor
            saliency = saliency_output.squeeze().cpu().numpy()
        else:
            # Fallback model already returns numpy
            saliency = saliency_output.squeeze().cpu().numpy()
    
    # Remove padding to match original aspect ratio
    h, w = orig_np.shape[:2]
    saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
    # Apply colormap and blend
    overlay, attention_score = apply_colormap_and_blend(orig_np, saliency)
    # Encode overlay as base64
    overlay_b64 = image_to_base64(overlay)
    return {
        "heatmap": overlay_b64,
        "attention_score": round(attention_score, 2),
        "model_type": "MIT1003-trained" if USE_TRAINED_MODEL else "VGG16-features"
    }

# --- Root Endpoint ---
@app.get("/")
def root():
    return {"message": "Saliency Prediction API. POST an image to /predict."}
