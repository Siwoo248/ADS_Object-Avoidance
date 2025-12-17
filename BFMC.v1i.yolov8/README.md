# Traffic Sign Detection Model

This directory contains scripts to train and run inference with a YOLO model for traffic sign detection and classification. The model is designed to detect 13 classes of traffic-related objects.

## Setup

1. Ensure you're in the virtual environment:
```bash
source venv/bin/activate
```

2. Install requirements if you haven't:
```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python train.py \
    --data data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0  # use "cpu" if no GPU available
```

When YOLOv12 weights become available, replace `yolov8n.pt` with the path to your YOLOv12 weights.

The training script will:
1. Validate the data configuration
2. Initialize the model
3. Train for the specified number of epochs
4. Save the best weights and training results

## Inference

To run inference on new images/video:

```bash
python infer.py \
    --model runs/train/traffic_signs_exp/weights/best.pt \
    --source /path/to/your/images \
    --conf 0.25 \
    --imgsz 640 \
    --device 0  # use "cpu" if no GPU available
```

Add `--save-txt` flag to save detection results in YOLO format.

## Classes

The model detects these 13 classes:
1. car
2. crosswalk
3. highway_entry
4. highway_exit
5. no_entry
6. onewayroad
7. parking
8. pedestrian
9. priority
10. roadblock
11. roundabout
12. stop
13. trafficlight

## Dataset Structure

The dataset should follow this structure:
```
traffic_sign_ads/BFMC.v1i.yolov8
├── train/
│   ├── images/
│   │   └── *.jpg/png
│   └── labels/
│       └── *.txt
├── valid/
│   ├── images/
│   │   └── *.jpg/png
│   └── labels/
│       └── *.txt
|
├── data.yaml
├── train.py
└── infer.py
```

Labels should be in YOLO format: one .txt file per image with each line containing:
`class_id center_x center_y width height` (normalized coordinates)
