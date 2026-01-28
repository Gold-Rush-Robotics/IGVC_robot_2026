#!/usr/bin/env python3
"""
YOLOv12 Training Script for IGVC Dataset
This script trains a YOLOv12 model on the IGVC dataset.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import argparse
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv12 model on IGVC dataset')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolov12n.pt',
        choices=['yolov12n.pt', 'yolov12s.pt', 'yolov12m.pt', 'yolov12l.pt', 'yolov12x.pt'],
        help='YOLOv12 model size (n=nano, s=small, m=medium, l=large, x=xlarge)'
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
        default='dataset/data.yaml',
        help='Path to dataset configuration file'
    )
    parser.add_argument(
        '--project', 
        type=str, 
        default='runs/detect',
        help='Project name (results directory)'
    )
    parser.add_argument(
        '--name', 
        type=str, 
        default=f'yolov12_igvc_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Experiment name'
    )
    parser.add_argument(
        '--cache', 
        action='store_true',
        help='Cache images for faster training'
    )
    parser.add_argument(
        '--amp', 
        action='store_true',
        default=True,
        help='Use Automatic Mixed Precision (AMP)'
    )
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
    
    # Print training configuration
    print("=" * 60)
    print("YOLOv12 Training Configuration")
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
    
    # Load YOLOv12 model
    print(f"Loading YOLOv12 model: {args.model}")
    model = YOLO(args.model)
    
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
        # Train model
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
