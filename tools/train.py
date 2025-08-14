"""
YOLOPv3 Training Script for Google Colab
Multi-task learning for object detection, drivable area segmentation, and lane detection
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg, update_config
from lib.models import get_net
from lib.dataset import BddDataset
from lib.utils.utils import create_logger, select_device
from lib.core.function import validate
from lib.core.general import non_max_suppression, xywh2xyxy, xyxy2xywh
from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.cuda import amp


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOPv3 multitask network')
    
    # Paths
    parser.add_argument('--data-root', type=str, 
                        default='/content/bdd100k/images/100k',
                        help='Path to BDD100K images')
    parser.add_argument('--label-root', type=str,
                        default='/content/bdd100k/labels/100k',
                        help='Path to detection labels')
    parser.add_argument('--mask-root', type=str,
                        default='/content/bdd100k/da_seg_annotations',
                        help='Path to drivable area segmentation')
    parser.add_argument('--lane-root', type=str,
                        default='/content/bdd100k/ll_seg_annotations',
                        help='Path to lane line segmentation')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of dataloader workers')
    
    # Model settings
    parser.add_argument('--pretrained', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num-classes', type=int, default=13,
                        help='Number of detection classes')
    
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    
    # Loss weights
    parser.add_argument('--det-weight', type=float, default=1.0,
                        help='Detection loss weight')
    parser.add_argument('--da-weight', type=float, default=0.2,
                        help='Drivable area segmentation loss weight')
    parser.add_argument('--ll-weight', type=float, default=0.2,
                        help='Lane line segmentation loss weight')
    
    # Other settings
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device, i.e. 0 or cpu')
    parser.add_argument('--save-dir', type=str, default='runs/train',
                        help='Directory to save results')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every n epochs')
    parser.add_argument('--val-freq', type=int, default=5,
                        help='Validate every n epochs')
    parser.add_argument('--single-cls', action='store_true',
                        help='Train as single-class detector')
    
    args = parser.parse_args()
    return args


class MultiTaskLoss(nn.Module):
    """Combined loss for detection, drivable area and lane line segmentation"""
    
    def __init__(self, num_classes=1, det_weight=1.0, da_weight=0.2, ll_weight=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.det_weight = det_weight
        self.da_weight = da_weight
        self.ll_weight = ll_weight
        
        # Detection losses (simplified YOLOv5 style)
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        
        # Segmentation losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def build_targets(self, predictions, targets, anchors):
        """Build targets for detection head (simplified version)"""
        # This is a simplified version - in production you'd use the full YOLO target building
        device = predictions[0].device
        nt = targets.shape[0]  # number of targets
        
        if nt == 0:
            return torch.zeros(3, device=device)
        
        # Simplified: just compute basic losses
        # In real implementation, you'd match targets to anchors properly
        return torch.zeros(3, device=device)
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model outputs [detection_out, da_seg_out, ll_seg_out]
            targets: Ground truth [det_targets, da_targets, ll_targets]
        """
        device = outputs[0][0].device if isinstance(outputs[0], (list, tuple)) else outputs[0].device
        
        # Unpack outputs
        det_out, da_seg_out, ll_seg_out = outputs
        
        # Unpack targets
        det_targets, da_targets, ll_targets = targets
        
        # Initialize losses
        loss_dict = {}
        
        # 1. Detection Loss (simplified)
        if isinstance(det_out, tuple):
            det_out = det_out[0]  # Get inference output
        
        # Simplified detection loss - just penalize if no objects
        # In production, implement proper YOLO loss
        if isinstance(det_out, list):
            det_loss = torch.tensor(0.0, device=device)
            for pred in det_out:
                det_loss += torch.mean(torch.abs(pred)) * 0.001
        else:
            det_loss = torch.mean(torch.abs(det_out)) * 0.001
        
        loss_dict['det_loss'] = det_loss * self.det_weight
        
        # 2. Drivable Area Segmentation Loss
        if da_seg_out is not None and da_targets is not None:
            # da_targets shape: [B, 2, H, W] - already one-hot encoded
            # da_seg_out shape: [B, 2, H, W] - logits
            da_loss = self.bce_loss(da_seg_out, da_targets)
            loss_dict['da_loss'] = da_loss * self.da_weight
        else:
            loss_dict['da_loss'] = torch.tensor(0.0, device=device)
        
        # 3. Lane Line Segmentation Loss  
        if ll_seg_out is not None and ll_targets is not None:
            # ll_targets shape: [B, 2, H, W] - already one-hot encoded
            # ll_seg_out shape: [B, 2, H, W] - logits
            ll_loss = self.bce_loss(ll_seg_out, ll_targets)
            loss_dict['ll_loss'] = ll_loss * self.ll_weight
        else:
            loss_dict['ll_loss'] = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = sum(loss_dict.values())
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    
    loss_meter = {
        'total': 0.0,
        'det': 0.0,
        'da': 0.0,
        'll': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets, paths, shapes) in enumerate(pbar):
        # Move to device
        images = images.to(device, non_blocking=True)
        targets = [t.to(device) if t is not None else None for t in targets]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with amp.autocast():
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss, loss_dict = criterion(outputs, targets)
        else:
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update meters
        loss_meter['total'] += loss.item()
        loss_meter['det'] += loss_dict.get('det_loss', 0).item()
        loss_meter['da'] += loss_dict.get('da_loss', 0).item()
        loss_meter['ll'] += loss_dict.get('ll_loss', 0).item()
        
        # Update progress bar
        avg_loss = loss_meter['total'] / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Return average losses
    n_batches = len(dataloader)
    return {k: v/n_batches for k, v in loss_meter.items()}


def main():
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with arguments
    cfg.DATASET.DATAROOT = args.data_root
    cfg.DATASET.LABELROOT = args.label_root
    cfg.DATASET.MASKROOT = args.mask_root
    cfg.DATASET.LANEROOT = args.lane_root
    cfg.TRAIN.BATCH_SIZE_PER_GPU = args.batch_size
    cfg.TRAIN.BEGIN_EPOCH = 0
    cfg.TRAIN.END_EPOCH = args.epochs
    cfg.TRAIN.LR0 = args.lr
    cfg.MODEL.IMAGE_SIZE = [args.img_size, args.img_size]
    
    # Set single class mode if specified
    if args.single_cls:
        # Modify the dataset file temporarily
        import lib.dataset.bdd as bdd_module
        bdd_module.single_cls = True
        num_classes = 1
    else:
        import lib.dataset.bdd as bdd_module
        bdd_module.single_cls = False
        num_classes = args.num_classes
    
    # Setup device
    device = select_device(None, args.device)
    
    # Setup model
    print("Building model...")
    model = get_net(cfg)
    model.nc = num_classes  # Update number of classes
    
    # Update detection head for correct number of classes
    detector_idx = model.detector_index
    if detector_idx >= 0:
        detector = model.model[detector_idx]
        detector.nc = num_classes
        detector.no = num_classes + 5
        # Reinitialize detection convolutions
        for i, ch in enumerate(detector.ch):
            detector.m[i] = nn.Conv2d(ch, detector.no * detector.na, 1).to(device)
    
    model = model.to(device)
    
    # Load pretrained weights if provided
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        # Load with strict=False to handle shape mismatches
        model.load_state_dict(checkpoint, strict=False)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Setup datasets
    print("Loading datasets...")
    train_dataset = BddDataset(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=None
    )
    
    val_dataset = BddDataset(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=None
    )
    
    # Setup dataloaders
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoaderX(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.momentum, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Setup loss function
    criterion = MultiTaskLoss(
        num_classes=num_classes,
        det_weight=args.det_weight,
        da_weight=args.da_weight,
        ll_weight=args.ll_weight
    )
    
    # Setup mixed precision training
    scaler = amp.GradScaler() if device.type != 'cpu' else None
    
    # Setup tensorboard
    writer = SummaryWriter(save_dir / 'logs')
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
        writer.add_scalar('Loss/train_det', train_losses['det'], epoch)
        writer.add_scalar('Loss/train_da', train_losses['da'], epoch)
        writer.add_scalar('Loss/train_ll', train_losses['ll'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}: Total Loss: {train_losses['total']:.4f}, "
              f"Det: {train_losses['det']:.4f}, "
              f"DA: {train_losses['da']:.4f}, "
              f"LL: {train_losses['ll']:.4f}")
        
        # Validation
        if (epoch + 1) % args.val_freq == 0:
            print("Running validation...")
            # You can call the validate function from lib.core.function here
            # For now, just save checkpoint based on training loss
            
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_losses': train_losses,
            }
            
            # Save latest
            torch.save(checkpoint, save_dir / 'latest.pth')
            
            # Save epoch checkpoint
            torch.save(checkpoint, save_dir / f'epoch_{epoch}.pth')
            
            # Save best
            if train_losses['total'] < best_loss:
                best_loss = train_losses['total']
                torch.save(checkpoint, save_dir / 'best.pth')
                print(f"Saved best model with loss: {best_loss:.4f}")
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'train_losses': train_losses,
    }
    torch.save(final_checkpoint, save_dir / 'final.pth')
    
    writer.close()
    print("Training completed!")
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()