import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import math

from dataset import DegradationDataset # Imports your dataset

# --------------------------------------------------------------------------- #
# Hardware-Aware Architecture (Depthwise-Separable Edge-UNet)
# --------------------------------------------------------------------------- #
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.pointwise(self.depthwise(x)))

class EdgeUNet(nn.Module):
    """Ultra-lightweight UNet designed for sub-50ms offline inference."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DepthwiseSeparableConv(1, 16)
        self.enc2 = DepthwiseSeparableConv(16, 32, stride=2)
        self.enc3 = DepthwiseSeparableConv(32, 64, stride=2)
        
        # Bottleneck
        self.bottle = DepthwiseSeparableConv(64, 64)
        
        # Decoder (Using bilinear upsampling to avoid checkerboard artifacts)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DepthwiseSeparableConv(64 + 32, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DepthwiseSeparableConv(32 + 16, 16)
        
        # Output
        self.final = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.bottle(e3)
        
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        
        # Bounding output strictly between 0 and 1
        return torch.sigmoid(self.final(d2))

# --------------------------------------------------------------------------- #
# Hybrid Loss (SSIM + L1)
# --------------------------------------------------------------------------- #
def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_loss(img1, img2, window_size=11):
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()

def hybrid_loss(pred, target):
    l1 = F.l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    return 0.1 * l1 + 0.9 * ssim

# --------------------------------------------------------------------------- #
# Training Loop
# --------------------------------------------------------------------------- #
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # TODO: Point to your clean data directory
    dataset = DegradationDataset(clean_dir="data/clean", patch_size=256, degrade=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    model = EdgeUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = GradScaler()

    os.makedirs("models", exist_ok=True)
    best_loss = float('inf')

    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for degraded, clean in loader:
            degraded, clean = degraded.to(device), clean.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                restored = model(degraded)
                loss = hybrid_loss(restored, clean)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/best_weights.pth")
            print(" -> Checkpoint saved!")

if __name__ == "__main__":
    train()
