#!/usr/bin/env python3
"""
YOLO26 training script for the IGVC dataset.
This script trains a YOLO26 model on the IGVC dataset.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import argparse
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_training_path(path_value):
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return SCRIPT_DIR / path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO26 model on IGVC dataset')
    
    parser.add_argument(
        '--model', 
        type=str, 
<<<<<<<< HEAD:training/train_yolov12.py
        default=str(SCRIPT_DIR / 'weights' / 'yolov12n.pt'),
        help='YOLOv12 model path or model size (yolov12n.pt, yolov12s.pt, etc.)'
========
        default='yolo26n.pt',
        choices=['yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt'],
        help='YOLO26 model size (n=nano, s=small, m=medium, l=large, x=xlarge)'
>>>>>>>> origin/main:train_yolov26.py
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='0',
        help='GPU device ID (e.g., "0", "0,1" for multiple GPUs, or "cpu")'
    )
    parser.add_argument(
        '--patience', 
        type=int, 
        default=20,
        help='Early stopping patience (epochs)'
    )
    parser.add_argument(
        '--lr0', 
        type=float, 
        default=0.01,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default=str(SCRIPT_DIR / 'dataset' / 'data.yaml'),
        help='Path to dataset configuration file'
    )
    parser.add_argument(
        '--project', 
        type=str, 
        default=str(SCRIPT_DIR / 'runs' / 'detect'),
        help='Project name (results directory)'
    )
    parser.add_argument(
        '--name', 
        type=str, 
        default=f'yolo26_igvc_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Experiment name'
    )
    parser.add_argument(
        '--cache', 
        action='store_true',
        help='Cache images for faster training'
    )
    parser.add_argument(
        '--amp', 
        dest='amp',
        action='store_true',
        help='Use Automatic Mixed Precision (AMP)'
    )
    parser.add_argument(
        '--no-amp',
        dest='amp',
        action='store_false',
        help='Disable Automatic Mixed Precision (AMP)'
    )
    parser.set_defaults(amp=True)
    parser.add_argument(
        '--hsv-h', 
        type=float, 
        default=0.015,
        help='HSV-Hue augmentation range'
    )
    parser.add_argument(
        '--hsv-s', 
        type=float, 
        default=0.7,
        help='HSV-Saturation augmentation range'
    )
    parser.add_argument(
        '--hsv-v', 
        type=float, 
        default=0.4,
        help='HSV-Value augmentation range'
    )
    parser.add_argument(
        '--degrees', 
        type=float, 
        default=10.0,
        help='Rotation degrees for augmentation'
    )
    parser.add_argument(
        '--translate', 
        type=float, 
        default=0.1,
        help='Image translation range for augmentation'
    )
    parser.add_argument(
        '--scale', 
        type=float, 
        default=0.5,
        help='Image scale range for augmentation'
    )
    parser.add_argument(
        '--flipud', 
        type=float, 
        default=0.0,
        help='Probability of flip upside-down'
    )
    parser.add_argument(
        '--fliplr', 
        type=float, 
        default=0.5,
        help='Probability of flip left-right'
    )
    parser.add_argument(
        '--mosaic', 
        type=float, 
        default=1.0,
        help='Mosaic augmentation probability'
    )
    
    return parser.parse_args()

def download_model(model_name):
    """Resolve the requested model weights.

    Ultralytics can fetch official weight names like ``yolo26n.pt`` on demand,
    so only local custom paths need to exist ahead of time.
    """
    model_path = Path(model_name).resolve()
    
    if model_path.exists():
        print(f"✓ Model found locally: {model_path}")
        return model_path

    print(f"ℹ Using Ultralytics model asset: {model_name}")
    return model_name

def check_dataset(data_path):
    """Verify dataset configuration and paths."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"❌ Dataset config not found: {data_path}")
        sys.exit(1)
    
    print(f"✓ Dataset config found: {data_path}")
    
    # Check that train/val/test directories exist
    with open(data_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    for split in ['train', 'val', 'test']:
        split_path = data_path.parent / config.get(split, f'../{split}/images')
        if split_path.exists():
            print(f"✓ {split.capitalize()} split found: {split_path}")
        else:
            print(f"⚠ {split.capitalize()} split not found: {split_path}")


def main():
    """Main training function."""
    args = parse_arguments()
    args.data = str(resolve_training_path(args.data))
    args.project = str(resolve_training_path(args.project))
    if '/' in args.model or '\\' in args.model:
        args.model = str(resolve_training_path(args.model))
    
    # Print training configuration
    print("=" * 60)
    print("YOLO26 Training Configuration")
    print("=" * 60)
    print(f"Model:              {args.model}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch Size:         {args.batch}")
    print(f"Image Size:         {args.imgsz}")
    print(f"Device:             {args.device}")
    print(f"Early Stopping:     {args.patience} epochs")
    print(f"Initial LR:         {args.lr0}")
    print(f"Data Config:        {args.data}")
    print(f"Project:            {args.project}")
    print(f"Experiment:         {args.name}")
    print(f"Cache Images:       {args.cache}")
    print(f"AMP:                {args.amp}")
    print(f"Augmentation:")
    print(f"  - HSV-H:          {args.hsv_h}")
    print(f"  - HSV-S:          {args.hsv_s}")
    print(f"  - HSV-V:          {args.hsv_v}")
    print(f"  - Rotation:       {args.degrees}°")
    print(f"  - Translation:    {args.translate}")
    print(f"  - Scale:          {args.scale}")
    print(f"  - Mosaic:         {args.mosaic}")
    print(f"  - Flip UD:        {args.flipud}")
    print(f"  - Flip LR:        {args.fliplr}")
    print("=" * 60)
    
    # Check dataset
    print("\nValidating dataset...")
    check_dataset(args.data)
    print()
    
    # Load YOLO26 model
    print(f"Loading YOLO26 model: {args.model}")
    model_source = download_model(args.model)
    model = YOLO(model_source)
    
    # Training configuration
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'patience': args.patience,
        'lr0': args.lr0,
        'cache': args.cache,
        'amp': args.amp,
        'project': args.project,
        'name': args.name,
        'save': True,
        'save_period': 10,
        'exist_ok': False,
        'pretrained': True,
        'optimizer': 'SGD',
        'close_mosaic': 15,  # Close mosaic augmentation in last 15 epochs
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'perspective': 0.0,
        'verbose': True,
        'seed': 42,
    }
    
    if args.resume:
        print("\n⚠ Resuming training from last checkpoint...\n")
        results = model.train(resume=True, **{k: v for k, v in train_params.items() if k != 'exist_ok'})
    else:
        print("\n▶ Starting training...\n")
        results = model.train(**train_params)
    
    # Print results summary
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Results saved to: {args.project}/{args.name}")
    print(f"Best model: {args.project}/{args.name}/weights/best.pt")
    print(f"Last model: {args.project}/{args.name}/weights/last.pt")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
