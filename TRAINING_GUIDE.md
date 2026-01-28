# YOLOv12 Training Guide for IGVC Dataset

This directory contains scripts to train a YOLOv12 object detection model on the IGVC dataset.

## Prerequisites

1. Install Python 3.8 or later
2. Install required packages:
   ```bash
   pip install -r training_requirements.txt
   ```

## Files

- **train_yolov12.py** - Main Python training script with full configuration options
- **train_yolov12.sh** - Convenient bash wrapper for quick training
- **training_requirements.txt** - Python dependencies
- **dataset/data.yaml** - Dataset configuration (train/val/test splits)

## Quick Start

### Option 1: Using the Bash Script (Easiest)

```bash
# Train with default settings (nano model, 100 epochs)
./train_yolov12.sh

# Train with medium model for 200 epochs
./train_yolov12.sh --model yolov12m.pt --epochs 200

# Train with specific batch size and GPU
./train_yolov12.sh --batch 32 --device 0

# Resume previous training
./train_yolov12.sh --resume

# Show all options
./train_yolov12.sh --help
```

### Option 2: Using Python Script (Full Control)

```bash
# Train with default settings
python3 train_yolov12.py

# Train large model with custom settings
python3 train_yolov12.py --model yolov12l.pt --epochs 300 --batch 32 --imgsz 640

# Use multiple GPUs
python3 train_yolov12.py --device 0,1,2,3

# Use CPU only
python3 train_yolov12.py --device cpu

# Resume training
python3 train_yolov12.py --resume

# View all options
python3 train_yolov12.py --help
```

## Training Parameters

### Model Selection
- `yolov12n.pt` - Nano (fastest, least accurate) - 1.3M parameters
- `yolov12s.pt` - Small - 3.3M parameters  
- `yolov12m.pt` - Medium - 9.3M parameters
- `yolov12l.pt` - Large - 20M parameters
- `yolov12x.pt` - XLarge (slowest, most accurate) - 55M parameters

### Common Options
- `--epochs` - Number of training iterations (default: 100)
- `--batch` - Batch size (default: 16). Increase for faster training but more memory
- `--imgsz` - Input image size (default: 640). Larger = more detail but slower
- `--device` - GPU ID (e.g., "0") or "cpu" (default: 0)
- `--patience` - Early stopping if no improvement for N epochs (default: 20)
- `--lr0` - Initial learning rate (default: 0.01)
- `--cache` - Cache images in memory for faster training (requires more RAM)

### Data Augmentation
- `--degrees` - Rotation angle range (default: 10)
- `--translate` - Translation range (default: 0.1)
- `--scale` - Scale range (default: 0.5)
- `--fliplr` - Horizontal flip probability (default: 0.5)
- `--flipud` - Vertical flip probability (default: 0.0)
- `--hsv-h` - HSV hue range (default: 0.015)
- `--hsv-s` - HSV saturation range (default: 0.7)
- `--hsv-v` - HSV value range (default: 0.4)
- `--mosaic` - Mosaic augmentation probability (default: 1.0)

## Training Examples

### Fast Training (for testing)
```bash
python3 train_yolov12.py --model yolov12n.pt --epochs 10 --batch 32
```

### Standard Training
```bash
python3 train_yolov12.py --model yolov12m.pt --epochs 100 --batch 16 --patience 20
```

### High-Quality Training
```bash
python3 train_yolov12.py --model yolov12l.pt --epochs 300 --batch 32 --imgsz 1280 --patience 30
```

### Multi-GPU Training
```bash
python3 train_yolov12.py --device 0,1 --batch 64 --epochs 200
```

## Output Structure

Training results are saved in `runs/detect/yolov12_igvc_YYYYMMDD_HHMMSS/`:

```
runs/detect/yolov12_igvc_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt      # Best model based on validation mAP
│   └── last.pt      # Last checkpoint
├── results.csv      # Training metrics
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── PR_curve.png
├── R_curve.png
├── results.png      # Training plots
└── train_batch*.jpg # Sample training batches
```

## Using Trained Models

### Inference on Images
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/yolov12_igvc_YYYYMMDD_HHMMSS/weights/best.pt')

# Predict on image
results = model.predict(source='image.jpg', conf=0.25)

# Display results
results[0].show()
```

### Inference on Video
```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolov12_igvc_YYYYMMDD_HHMMSS/weights/best.pt')
results = model.predict(source='video.mp4', conf=0.25)
```

## Hardware Requirements

### Minimum (Nano model)
- GPU: 2GB VRAM (can use CPU)
- RAM: 4GB
- Time: ~1 hour per 100 epochs

### Recommended (Medium model)
- GPU: 8GB VRAM (NVIDIA RTX 3060+)
- RAM: 16GB
- Time: ~2-3 hours per 100 epochs

### High-Performance (Large model)
- GPU: 24GB VRAM (NVIDIA RTX 4090 or better)
- RAM: 32GB
- Time: ~3-5 hours per 100 epochs

## Tips for Best Results

1. **Start Small**: Begin with nano or small model for testing, then scale up
2. **Monitor Training**: Watch the training curves in TensorBoard
3. **Adjust Batch Size**: Increase batch size if GPU memory allows
4. **Data Augmentation**: The default settings work well; adjust if underfitting/overfitting
5. **Early Stopping**: The `--patience` parameter stops training if no improvement
6. **Resume Training**: If interrupted, use `--resume` to continue
7. **Validation**: Check results on test set after training completes

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 512`
- Use smaller model: `--model yolov12n.pt`
- Use CPU: `--device cpu` (slower)

### Training Too Slow
- Increase batch size: `--batch 64`
- Enable image caching: `--cache`
- Use multiple GPUs: `--device 0,1`
- Use smaller image size: `--imgsz 512`

### Model Not Improving
- Increase training time: `--epochs 200` or higher
- Use larger model: `--model yolov12l.pt`
- Adjust learning rate: `--lr0 0.001`
- Reduce augmentation: lower `--scale`, `--degrees`, etc.

## References

- YOLOv12 Documentation: https://docs.ultralytics.com/
- YOLOv12 GitHub: https://github.com/ultralytics/ultralytics
- IGVC Dataset: See dataset/README.roboflow.txt
