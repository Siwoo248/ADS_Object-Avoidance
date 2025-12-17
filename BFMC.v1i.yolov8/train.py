#!/usr/bin/env python3
"""Train a YOLO model for traffic sign detection and classification.

This script uses the ultralytics YOLO API and works with the traffic sign dataset
that has 13 classes including car, crosswalk, highway_entry, etc.
"""
import argparse
from pathlib import Path
import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLO model for traffic signs")
    p.add_argument("--model", default="yolov8m.pt",
                   help="Path to model weights (use yolov12 weights when available)")
    p.add_argument("--data", type=str, required=True,
                   help="Path to data.yaml file")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0",
                   help="Device to train on (e.g., cuda device='0' or 'cpu')")
    p.add_argument("--name", default="traffic_signs_exp",
                   help="Name for this experiment")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from last checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate data.yaml exists and has required fields
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config {args.data} not found")

    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    required_keys = ['train', 'val', 'nc', 'names']
    missing = [k for k in required_keys if k not in data_cfg]
    if missing:
        raise ValueError(f"Missing required keys in {args.data}: {missing}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not found. Install with: pip install ultralytics")

    # Initialize model - if custom YOLOv12 weights provided, they'll be used here
    model = YOLO(args.model)

    # Start training
    print(f"Starting training with:")
    print(f"- Model: {args.model}")
    print(f"- Data: {args.data}")
    print(f"- Image size: {args.imgsz}x{args.imgsz}")
    print(f"- Batch size: {args.batch}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Device: {args.device}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
