# YOLO Object Detection for Road Signs and Traffic Objects

This repository contains a YOLOv8-based object detection model trained for road sign and traffic object detection and classification. The model can detect and classify 13 different classes of traffic signs and objects commonly found in road environments.

## 🚀 Features

- **Real-time Detection**: Fast inference using YOLOv8 architecture
- **13 Classes**: Comprehensive coverage of traffic signs and objects
- **High Accuracy**: Trained on large dataset with 13,761 training images
- **Easy Integration**: Simple Python API for inference
- **GPU Support**: Optimized for GPU acceleration

## 📊 Model Classes

The model detects the following 13 classes:

1. **car** - Vehicles on the road
2. **crosswalk** - Pedestrian crossing signs
3. **highway_entry** - Highway entrance signs
4. **highway_exit** - Highway exit signs
5. **no_entry** - No entry signs
6. **onewayroad** - One-way road signs
7. **parking** - Parking signs
8. **pedestrian** - Pedestrians
9. **priority** - Priority road signs
10. **roadblock** - Road block signs
11. **roundabout** - Roundabout signs
12. **stop** - Stop signs
13. **trafficlight** - Traffic lights

## 🛠 Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/ADS-Skynet/Yolo_object_detection.git
cd yolo-object-detection

# Install dependencies
pip install ultralytics opencv-python numpy

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Training Details

### Dataset
- **Source**: BFMC.v1i.yolov8 dataset from Roboflow
- **Training Images**: 13,761
- **Validation Images**: 1,277
- **Test Images**: 1,277
- **Image Size**: 640x640 pixels

### Model Configuration
- **Architecture**: YOLOv8s (Small)
- **Epochs**: 100
- **Batch Size**: 8
- **Image Size**: 640x640
- **Device**: GPU (NVIDIA GeForce RTX 3050)

### Training Results
- **Best mAP@0.5**: 0.977
- **Best mAP@0.5:0.95**: 0.864
- **Training Time**: ~8 hours on GPU

## 🚀 Usage

### Python Inference
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('path/to/best.pt')

# Run inference on an image
results = model.predict('path/to/image.jpg', conf=0.25, save=True)

# Run inference on video
results = model.predict('path/to/video.mp4', save=True)

# Run inference on webcam
results = model.predict(source=0, show=True)
```

### Command Line
```bash
# Single image
yolo predict model=path/to/best.pt source=path/to/image.jpg

# Multiple images
yolo predict model=path/to/best.pt source=path/to/images/ save=True

# Video
yolo predict model=path/to/best.pt source=path/to/video.mp4 save=True
```

## 📊 Performance Metrics

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|----------|---------------|
| car | 0.95 | 0.92 | 0.94 | 0.78 |
| crosswalk | 0.98 | 0.96 | 0.97 | 0.85 |
| highway_entry | 0.97 | 0.95 | 0.96 | 0.82 |
| ... | ... | ... | ... | ... |

*Detailed metrics available in the training results folder*

## 📁 Project Structure

```
yolo_object_detection/
├── BFMC.v1i.yolov8/           # Dataset and training files
│   ├── data.yaml              # Dataset configuration
│   ├── train.py               # Training script
│   ├── runs/                  # Training results
│   │   └── detect/
│   │       └── traffic_signs_full_ads/
│   │           ├── weights/
│   │           │   ├── best.pt    # Best model weights
│   │           │   └── last.pt    # Last epoch weights
│   │           ├── results.csv    # Training metrics
│   │           └── *.png         # Training plots
│   ├── test/                   # Test dataset
│   ├── train/                  # Training dataset
│   └── valid/                  # Validation dataset
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## 🔧 Model Download

The trained model weights are available in the `BFMC.v1i.yolov8/runs/detect/traffic_signs_full_ads/weights/` directory:

- `best.pt`: Best performing model (recommended)
- `last.pt`: Model from the final training epoch

## 🧪 Testing

Test the model on sample images:

```bash
cd BFMC.v1i.yolov8
python -c "
from ultralytics import YOLO
import glob
model = YOLO('runs/detect/traffic_signs_full_ads/weights/best.pt')
images = glob.glob('test1/*.jpg')
results = model.predict(source=images, conf=0.25, save=True)
"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: BFMC.v1i.yolov8 from Roboflow
- **Framework**: Ultralytics YOLOv8
- **Training**: Performed on NVIDIA GeForce RTX 3050 GPU

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This model is trained specifically for traffic sign and object detection in road environments. Performance may vary in different lighting conditions or with objects not present in the training dataset.