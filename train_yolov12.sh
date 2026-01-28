#!/bin/bash
# YOLOv12 Training Script Wrapper
# Usage: ./train_yolov12.sh [options]

# Default values
MODEL="yolov12n.pt"
EPOCHS=100
BATCH=128
IMGSZ=640
DEVICE="0"
PATIENCE=20
RESUME=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --help)
            echo "YOLOv12 Training Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model size (yolov12n.pt, yolov12s.pt, etc.) [default: yolov12n.pt]"
            echo "  --epochs N          Number of epochs [default: 100]"
            echo "  --batch N           Batch size [default: 16]"
            echo "  --imgsz N           Image size [default: 640]"
            echo "  --device ID         GPU device ID [default: 0]"
            echo "  --patience N        Early stopping patience [default: 20]"
            echo "  --resume            Resume from last checkpoint"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./train_yolov12.sh                                  # Train with defaults"
            echo "  ./train_yolov12.sh --model yolov12m.pt --epochs 200 # Train medium model for 200 epochs"
            echo "  ./train_yolov12.sh --resume                         # Resume last training"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build Python command
CMD="python3 train_yolov12.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch $BATCH"
CMD="$CMD --imgsz $IMGSZ"
CMD="$CMD --device $DEVICE"
CMD="$CMD --patience $PATIENCE"

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

echo "Running: $CMD"
echo ""

# Execute training
eval $CMD
