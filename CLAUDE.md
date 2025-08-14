# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOPv3 is a multi-task learning model for autonomous driving that performs:
- Traffic object detection
- Drivable area segmentation  
- Lane line detection

The model uses an ELAN-Net backbone and is trained on the BDD100K dataset.

## Key Commands

### Testing/Evaluation
```bash
python tools/test.py --weights weights/epoch-189.pth --conf_thres 0.001 --iou_thres 0.6
```

### Demo/Inference
```bash
# For images
python tools/demo.py --weights weights/epoch-189.pth \
                     --source inference/image \
                     --save-dir inference/image_output \
                     --conf-thres 0.3 \
                     --iou-thres 0.45

# For video
python tools/demo.py --weights weights/epoch-189.pth \
                     --source path/to/video.mp4 \
                     --save-dir inference/video_output
```

### Training
Training code is not yet available in the repository (mentioned as "coming soon" in README).

## Architecture Overview

### Core Components

- **lib/models/YOLOP.py**: Main model architecture combining detection and segmentation heads
- **lib/models/common.py**: Building blocks and computation modules (ELAN-Net components)
- **lib/core/function.py**: Training and validation logic
- **lib/core/evaluate.py**: Metric calculations for all three tasks
- **lib/dataset/bdd.py**: BDD100K dataset implementation with multi-task loading

### Loss Functions
- **lib/core/loss.py**: Multi-task loss combining detection, drivable area, and lane line losses
- Loss weights configured in `lib/config/default.py`:
  - Detection: BOX_GAIN=0.05, CLS_GAIN=0.5, OBJ_GAIN=1.0
  - Drivable area: DA_SEG_GAIN=0.2
  - Lane line: LL_SEG_GAIN=0.2, LL_IOU_GAIN=0.2

### Configuration
Main configuration file: **lib/config/default.py**
- Model image size: [640, 640]
- Dataset paths need to be updated for your local setup
- Training parameters: batch_size=8, epochs=200, optimizer='adamw'

## Dataset Structure

The BDD100K dataset should be organized as:
```
dataset_root/
├── images/
│   ├── train/
│   └── val/
├── det_annotations/
│   ├── train/
│   └── val/
├── da_seg_annotations/
│   ├── train/
│   └── val/
└── ll_seg_annotations/
    ├── train/
    └── val/
```

Update dataset paths in `lib/config/default.py`:
- DATASET.DATAROOT: images folder
- DATASET.LABELROOT: detection annotations
- DATASET.MASKROOT: drivable area segmentation
- DATASET.LANEROOT: lane line segmentation

## Dependencies

Core requirements:
- Python 3.7+
- PyTorch 1.12+ with CUDA 11.3
- torchvision 0.13+
- OpenCV, Pillow, matplotlib
- tensorboardX for logging
- albumentations for augmentation

Install with: `pip install -r requirements.txt`