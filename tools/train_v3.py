"""
YOLOPv3 Training Script - Based on YOLOPv1 Structure
Complete training implementation with all features from YOLOPv1
"""

import argparse
import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter
import lib.dataset as dataset
from lib.config import cfg, update_config
from lib.core.loss import get_loss
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import create_logger, select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOPv3 multitask network')
    
    # Paths
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    
    # Dataset paths (for Colab)
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
    
    # Training mode
    parser.add_argument('--det-only', action='store_true', help='Train detection only')
    parser.add_argument('--seg-only', action='store_true', help='Train segmentation only')
    parser.add_argument('--lane-only', action='store_true', help='Train lane detection only')
    parser.add_argument('--drivable-only', action='store_true', help='Train drivable area only')
    
    # Multi-class settings
    parser.add_argument('--single-cls', action='store_true', help='Train as single-class detector')
    parser.add_argument('--num-classes', type=int, default=13, help='Number of detection classes')
    
    # DDP settings
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    
    # NMS settings
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='0', help='CUDA device')
    
    # Model settings
    parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--autoanchor', action='store_true', help='Re-compute anchors')
    
    args = parser.parse_args()
    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(cfg, model):
    """Get optimizer based on config"""
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)  # apply decay
    
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(g0, lr=cfg.TRAIN.LR0, betas=(cfg.TRAIN.MOMENTUM, 0.999))
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(g0, lr=cfg.TRAIN.LR0, betas=(cfg.TRAIN.MOMENTUM, 0.999))
    else:
        optimizer = torch.optim.SGD(g0, lr=cfg.TRAIN.LR0, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    
    optimizer.add_param_group({'params': g1, 'weight_decay': cfg.TRAIN.WD})
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    
    return optimizer


def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
          writer_dict, logger, device, rank=-1):
    """Train for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Component losses
    loss_meters = {
        'box': AverageMeter(),
        'obj': AverageMeter(),
        'cls': AverageMeter(),
        'da_seg': AverageMeter(),
        'll_seg': AverageMeter(),
        'll_iou': AverageMeter()
    }
    
    # Switch to train mode
    model.train()
    start = time.time()
    
    pbar = enumerate(train_loader)
    if rank in [-1, 0]:
        pbar = tqdm(pbar, total=len(train_loader), desc=f'Epoch {epoch}/{cfg.TRAIN.END_EPOCH}')
    
    for i, (input, target, paths, shapes) in pbar:
        num_iter = i + num_batch * (epoch - 1)
        
        # Warmup
        if num_iter < num_warmup:
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])
        
        data_time.update(time.time() - start)
        
        # Move data to device
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                if tgt is not None:
                    assign_target.append(tgt.to(device))
                else:
                    assign_target.append(None)
            target = assign_target
        
        # Forward pass with mixed precision
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, shapes, model)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        if rank in [-1, 0]:
            losses.update(total_loss.item(), input.size(0))
            
            # Update component losses
            for key in loss_meters:
                loss_key = 'l' + key.replace('_', '')
                if loss_key in head_losses:
                    loss_meters[key].update(head_losses[loss_key].item(), input.size(0))
            
            # Update progress bar
            batch_time.update(time.time() - start)
            
            if i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                          epoch, i, num_batch,
                          batch_time=batch_time,
                          speed=input.size(0) / batch_time.val,
                          data_time=data_time,
                          loss=losses)
                
                # Add component losses
                loss_str = ' '.join([f'{k}:{v.avg:.4f}' for k, v in loss_meters.items() if v.count > 0])
                msg += loss_str
                logger.info(msg)
                
                # Write to tensorboard
                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', losses.avg, global_steps)
                    for key, meter in loss_meters.items():
                        if meter.count > 0:
                            writer.add_scalar(f'train_{key}_loss', meter.avg, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1
        
        start = time.time()
    
    return losses.avg


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    """Save checkpoint"""
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'model_best.pth'))


def main():
    args = parse_args()
    update_config(cfg, args)
    
    # Make config mutable to allow updates
    cfg.defrost()
    
    # Update config with dataset paths
    if args.data_root:
        cfg.DATASET.DATAROOT = args.data_root
    if args.label_root:
        cfg.DATASET.LABELROOT = args.label_root
    if args.mask_root:
        cfg.DATASET.MASKROOT = args.mask_root
    if args.lane_root:
        cfg.DATASET.LANEROOT = args.lane_root
    
    # Update training mode
    cfg.TRAIN.DET_ONLY = args.det_only
    cfg.TRAIN.SEG_ONLY = args.seg_only
    cfg.TRAIN.LANE_ONLY = args.lane_only
    cfg.TRAIN.DRIVABLE_ONLY = args.drivable_only
    
    # Update batch size and epochs
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE_PER_GPU = args.batch_size
    if args.epochs:
        cfg.TRAIN.END_EPOCH = args.epochs
    
    # Freeze config after updates
    cfg.freeze()
    
    # Set single class mode
    if args.single_cls:
        import lib.dataset.bdd as bdd_module
        bdd_module.single_cls = True
        num_classes = 1
    else:
        import lib.dataset.bdd as bdd_module
        bdd_module.single_cls = False
        num_classes = args.num_classes
    
    # DDP settings
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    rank = global_rank
    
    # Create logger
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)
    
    if rank in [-1, 0]:
        logger.info(f"Training YOLOPv3 with {num_classes} classes")
        logger.info(cfg)
        
        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None
    
    # CUDNN settings
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    # Setup device
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) \
        if not cfg.DEBUG else select_device(logger, 'cpu')
    
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    
    # Build model
    logger.info("Building model...")
    model = get_net(cfg).to(device)
    model.nc = num_classes
    
    # Update detection head for correct number of classes
    detector_idx = model.detector_index
    if detector_idx >= 0:
        detector = model.model[detector_idx]
        detector.nc = num_classes
        detector.no = num_classes + 5
        # Reinitialize detection convolutions
        for i, ch in enumerate(detector.ch):
            detector.m[i] = nn.Conv2d(ch, detector.no * detector.na, 1).to(device)
    
    # Define loss and optimizer
    criterion = get_loss(cfg, device=device)
    optimizer = get_optimizer(cfg, model)
    
    # Load checkpoint
    best_perf = 0.0
    last_epoch = -1
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    
    # Learning rate scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Load pretrained model
    if cfg.MODEL.PRETRAINED and os.path.exists(cfg.MODEL.PRETRAINED):
        logger.info(f"Loading pretrained model from {cfg.MODEL.PRETRAINED}")
        checkpoint = torch.load(cfg.MODEL.PRETRAINED, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        if 'epoch' in checkpoint:
            begin_epoch = checkpoint['epoch']
    
    # Resume training
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')
    if args.resume and os.path.exists(checkpoint_file):
        logger.info(f"Resuming from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info(f"Resumed from epoch {begin_epoch}")
    
    # Setup model layers for selective training
    Encoder_para_idx = [str(i) for i in range(0, 2)]  # Backbone
    Det_Head_para_idx = [str(i) for i in range(2, 4)]  # Detection head
    Da_Seg_Head_para_idx = [str(i) for i in range(4, 18)]  # DA segmentation
    Ll_Seg_Head_para_idx = [str(i) for i in range(18, 28)]  # Lane line segmentation
    
    # Freeze layers based on training mode
    if cfg.TRAIN.SEG_ONLY:
        logger.info('Freezing encoder and detection head...')
        for k, v in model.named_parameters():
            v.requires_grad = True
            if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                v.requires_grad = False
    
    if cfg.TRAIN.DET_ONLY:
        logger.info('Freezing segmentation heads...')
        for k, v in model.named_parameters():
            v.requires_grad = True
            if k.split(".")[1] in Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                v.requires_grad = False
    
    if cfg.TRAIN.LANE_ONLY:
        logger.info('Freezing encoder, detection and DA segmentation...')
        for k, v in model.named_parameters():
            v.requires_grad = True
            if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx + Da_Seg_Head_para_idx:
                v.requires_grad = False
    
    if cfg.TRAIN.DRIVABLE_ONLY:
        logger.info('Freezing encoder, detection and lane line segmentation...')
        for k, v in model.named_parameters():
            v.requires_grad = True
            if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx + Ll_Seg_Head_para_idx:
                v.requires_grad = False
    
    # DDP model
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    # Model parameters
    model.gr = 1.0
    model.nc = num_classes
    
    # Data loading
    logger.info("Loading datasets...")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
    
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    num_batch = len(train_loader)
    
    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        
        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
        logger.info(f'Train samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}')
    
    # Training
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    logger.info('Starting training...')
    logger.info(f'Training for {cfg.TRAIN.END_EPOCH} epochs with {num_classes} classes')
    
    for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train(cfg, train_loader, model, criterion, optimizer, scaler,
                          epoch, num_batch, num_warmup, writer_dict, logger, device, rank)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Validation
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
            logger.info('Running validation...')
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                logger, device, rank
            )
            
            # Calculate fitness
            fi = fitness(np.array(detect_results).reshape(1, -1))
            
            # Log results
            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                  'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                  'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                  'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                      epoch, loss=total_loss,
                      da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
                      ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
                      p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
                      t_inf=times[0], t_nms=times[1])
            logger.info(msg)
            
            # Save checkpoint
            if rank in [-1, 0]:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': total_loss,
                    'cfg': cfg,
                    'num_classes': num_classes
                }
                
                save_checkpoint(checkpoint, fi > best_perf, final_output_dir)
                
                if fi > best_perf:
                    best_perf = fi
                    logger.info(f'New best model with fitness: {fi:.4f}')
    
    # Save final model
    if rank in [-1, 0]:
        final_model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(final_model_state, os.path.join(final_output_dir, 'final_model.pth'))
        logger.info(f'Training completed. Final model saved to {final_output_dir}')
        
        if writer_dict:
            writer_dict['writer'].close()


if __name__ == '__main__':
    main()