#!/usr/bin/env python3
"""
Obstacle Avoidance — Imitation Learning Training Script
========================================================
Architecture:
  color (3ch)  → MobileNetV3-Small backbone (pretrained) → 576-dim
  depth (1ch)  → small CNN (3 conv layers)               →  64-dim
  mode (int)   → Embedding(4, 8)                         →   8-dim
                            ↓
               concat → [648-dim]
                            ↓
               MLP: 648 → 256 → 64 → 1  (steering)

Dataset:  data/images/*.jpg  +  data/images/*.npy  +  data/labels.csv
Modes collected: MICRO_ADJUST (1), LANE_CHANGE (2)  [NORMAL frames not saved]
Target:   input_steering  ∈ {-0.9, 0.0, 0.9}
"""

import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"
SAVE_PATH  = SCRIPT_DIR / "obstacle_avoidance_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Mode encoding  (NORMAL=0 kept for embedding completeness; not in this dataset)
# ─────────────────────────────────────────────────────────────────────────────
MODE_MAP = {
    'NORMAL':       0,
    'MICRO_ADJUST': 1,
    'LANE_CHANGE':  2,
    'STOP':         3,
}

# Depth normalisation constant: 10 000 raw units ≈ 10 m for RealSense D455
# (depth_scale ≈ 0.001 → 10 000 * 0.001 = 10 m)
DEPTH_NORM = 10_000.0

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ObstacleDataset(Dataset):
    """
    Each sample: (color_tensor, depth_tensor, mode_int, steering_float)
      color_tensor : (3, 224, 224)  float32, ImageNet-normalised
      depth_tensor : (1, 224, 224)  float32, normalised to [0, 1] (10 m = 1.0)
      mode_int     : long scalar    1=MICRO_ADJUST, 2=LANE_CHANGE
      steering     : float scalar   ∈ {-0.9, 0.0, 0.9}
    """

    def __init__(self, labels_csv: Path, images_dir: Path):
        # CSV has no header row
        self.df = pd.read_csv(
            labels_csv,
            header=None,
            names=["frame_id", "input_steering", "input_throttle", "mode"],
        )
        # Keep only the two labelled modes (NORMAL frames were never saved)
        self.df = self.df[
            self.df["mode"].isin(["MICRO_ADJUST", "LANE_CHANGE"])
        ].reset_index(drop=True)

        self.images_dir = images_dir

        self.color_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        frame_id = str(int(row["frame_id"])).zfill(6)

        # ── Color ─────────────────────────────────────────────────────────────
        img_bgr = cv2.imread(str(self.images_dir / f"{frame_id}.jpg"))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)          # (H, W, 3) uint8
        color   = self.color_transform(img_rgb)                      # (3, 224, 224)

        # ── Depth ─────────────────────────────────────────────────────────────
        depth_raw = np.load(str(self.images_dir / f"{frame_id}.npy"))
        if depth_raw.size == 0:
            depth_f32 = np.zeros((224, 224), dtype=np.float32)
        else:
            depth_f32 = depth_raw.astype(np.float32) / DEPTH_NORM
            depth_f32 = np.clip(depth_f32, 0.0, 1.0)
            depth_f32 = cv2.resize(depth_f32, (224, 224),
                                   interpolation=cv2.INTER_NEAREST)
        depth = torch.tensor(depth_f32, dtype=torch.float32).unsqueeze(0)  # (1, 224, 224)

        # ── Mode ──────────────────────────────────────────────────────────────
        mode = torch.tensor(MODE_MAP.get(row["mode"], 0), dtype=torch.long)

        # ── Target ────────────────────────────────────────────────────────────
        steering = torch.tensor(float(row["input_steering"]), dtype=torch.float32)

        return color, depth, mode, steering


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class DepthCNN(nn.Module):
    """1-channel depth image → 64-dim feature vector."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # → 16 × 112 × 112
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # → 32 ×  56 ×  56
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 64 ×  28 ×  28
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                     # → 64 ×   1 ×   1
            nn.Flatten(),                                # → 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObstacleAvoidanceModel(nn.Module):
    """
    Multi-modal model: color + depth + mode → steering angle.

    Feature dimensions:
      color branch : 576  (MobileNetV3-Small)
      depth branch :  64  (DepthCNN)
      mode branch  :   8  (Embedding)
      concat       : 648
      MLP output   :   1  (steering)
    """

    def __init__(self):
        super().__init__()

        # Color branch
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.color_features = backbone.features       # output: (B, 576, 7, 7)
        self.color_pool     = nn.AdaptiveAvgPool2d(1) # → (B, 576, 1, 1)
        color_dim = 576

        # Depth branch
        self.depth_cnn = DepthCNN()
        depth_dim = 64

        # Mode branch
        self.mode_embed = nn.Embedding(4, 8)  # modes 0-3
        mode_dim = 8

        # MLP
        fused_dim = color_dim + depth_dim + mode_dim  # 648
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        color: torch.Tensor,  # (B, 3, 224, 224)
        depth: torch.Tensor,  # (B, 1, 224, 224)
        mode:  torch.Tensor,  # (B,) long
    ) -> torch.Tensor:        # (B,)

        c = self.color_features(color)   # (B, 576, 7, 7)
        c = self.color_pool(c)           # (B, 576, 1, 1)
        c = c.flatten(1)                 # (B, 576)

        d = self.depth_cnn(depth)        # (B, 64)

        m = self.mode_embed(mode)        # (B, 8)

        x   = torch.cat([c, d, m], dim=1)  # (B, 648)
        out = self.mlp(x)                  # (B, 1)
        return out.squeeze(1)              # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train() -> None:
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        free, total = [x / 1024**2 for x in torch.cuda.mem_get_info(0)]
        print(f"VRAM   : {free:.0f} MB free / {total:.0f} MB total")
    print()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = ObstacleDataset(LABELS_CSV, IMAGES_DIR)

    print("=" * 50)
    print("  Dataset Verification")
    print("=" * 50)
    print(f"Total samples : {len(dataset)}")
    print()
    print("Mode distribution:")
    print(dataset.df["mode"].value_counts().to_string())
    print()
    print("Steering distribution:")
    print(dataset.df["input_steering"].value_counts().sort_index().to_string())
    print()

    # Sample shape check
    color0, depth0, mode0, steer0 = dataset[0]
    print("Sample shapes (index 0):")
    print(f"  color    : {tuple(color0.shape)}  dtype={color0.dtype}")
    print(f"  depth    : {tuple(depth0.shape)}  dtype={depth0.dtype}")
    print(f"  mode     : {mode0.item()}  ({dataset.df.iloc[0]['mode']})")
    print(f"  steering : {steer0.item():.4f}")
    print()

    if len(dataset) == 0:
        print("[ERROR] Dataset is empty — check labels.csv and images directory.")
        sys.exit(1)

    # ── Train / Val split ─────────────────────────────────────────────────────
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train / Val  : {n_train} / {n_val}")
    print()

    # batch_size=16, num_workers=0: Jetson Orin has unified CPU/GPU memory.
    # batch_size=32 exhausts the ~2.4 GB free during backward pass.
    # num_workers > 0 spawns subprocesses that each map GPU memory, making it worse.
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ObstacleAvoidanceModel().to(device)

    # ── Forward pass test ─────────────────────────────────────────────────────
    print("=" * 50)
    print("  Forward Pass Test")
    print("=" * 50)
    model.eval()
    with torch.no_grad():
        c_t = color0.unsqueeze(0).to(device)
        d_t = depth0.unsqueeze(0).to(device)
        m_t = mode0.unsqueeze(0).to(device)
        out = model(c_t, d_t, m_t)
    print(f"  color input  : {tuple(c_t.shape)}")
    print(f"  depth input  : {tuple(d_t.shape)}")
    print(f"  mode input   : {tuple(m_t.shape)}")
    print(f"  output       : {tuple(out.shape)}  value={out.item():.4f}")
    print("  PASSED\n")

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    print("=" * 50)
    print("  Training  (50 epochs, Adam lr=1e-4, MSELoss)")
    print("=" * 50)
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}")
    print("-" * 36)

    for epoch in range(1, 51):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for color, depth, mode, steering in train_loader:
            color    = color.to(device)
            depth    = depth.to(device)
            mode     = mode.to(device)
            steering = steering.to(device)

            optimizer.zero_grad()
            pred = model(color, depth, mode)
            loss = criterion(pred, steering)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(steering)

        train_loss = running_loss / n_train

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for color, depth, mode, steering in val_loader:
                color    = color.to(device)
                depth    = depth.to(device)
                mode     = mode.to(device)
                steering = steering.to(device)
                pred     = model(color, depth, mode)
                loss     = criterion(pred, steering)
                running_loss += loss.item() * len(steering)

        val_loss = running_loss / n_val

        # ── Save best ─────────────────────────────────────────────────────────
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            marker = "  *"

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}{marker}")

    print()
    print(f"Best val loss : {best_val_loss:.6f}")
    print(f"Model saved   → {SAVE_PATH}")


if __name__ == "__main__":
    train()
