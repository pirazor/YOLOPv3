import torch.nn as nn
import torch
from .general import bbox_iou
from .postprocess import build_targets
from lib.core.evaluate import SegmentationMetric


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class MultiHeadLoss(nn.Module):
    """
    Collect all the loss for multi-task learning
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model: model object
        - shapes: image shapes

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (dict) contain all loss components
        """
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)
        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """
        Args:
            predictions: predicts of [[det_head1, det_head2, ...], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model: model object
            shapes: image shapes

        Returns:
            total_loss: sum of all the loss
            head_losses: dict containing losses
        """
        cfg = self.cfg
        device = targets[0].device
        
        # Initialize losses
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        
        # Build targets for detection
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)
        
        # Class label smoothing
        cp, cn = smooth_BCE(eps=0.0)
        
        # Get loss functions
        BCEcls, BCEobj, BCEseg = self.losses
        
        # Calculate detection losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs (detection layers)
        balance = [4.0, 1.0, 0.4, 0.1, 0.1] if no == 5 else [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]
        
        # Detection loss calculation
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Regression loss
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)
                
                # Classification
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
                    
            # Objectness loss
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]
        
        # Segmentation losses
        # Drivable area segmentation loss
        if predictions[1] is not None and targets[1] is not None:
            drive_area_seg_predicts = predictions[1].view(-1)
            drive_area_seg_targets = targets[1].view(-1)
            lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)
        else:
            lseg_da = torch.zeros(1, device=device)
        
        # Lane line segmentation loss
        if predictions[2] is not None and targets[2] is not None:
            lane_line_seg_predicts = predictions[2].view(-1)
            lane_line_seg_targets = targets[2].view(-1)
            lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)
            
            # Lane line IoU loss
            metric = SegmentationMetric(2)
            nb, _, height, width = targets[2].shape
            pad_w, pad_h = shapes[0][1][1] if shapes else (0, 0)
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            
            _, lane_line_pred = torch.max(predictions[2], 1)
            _, lane_line_gt = torch.max(targets[2], 1)
            
            if pad_h > 0 and pad_w > 0:
                lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
                lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
            
            metric.reset()
            metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
            IoU = metric.IntersectionOverUnion()
            liou_ll = 1 - IoU
        else:
            lseg_ll = torch.zeros(1, device=device)
            liou_ll = torch.zeros(1, device=device)
        
        # Apply loss gains
        s = 3 / no  # output count scaling
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]
        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]
        
        # Handle training mode settings
        if hasattr(cfg.TRAIN, 'DET_ONLY') and cfg.TRAIN.DET_ONLY:
            lseg_da = 0 * lseg_da
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            
        if hasattr(cfg.TRAIN, 'SEG_ONLY') and cfg.TRAIN.SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            
        if hasattr(cfg.TRAIN, 'LANE_ONLY') and cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da
            
        if hasattr(cfg.TRAIN, 'DRIVABLE_ONLY') and cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
        
        # Total loss
        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
        
        # Return loss dictionary
        loss_dict = {
            'total_loss': loss,
            'lbox': lbox,
            'lobj': lobj,
            'lcls': lcls,
            'lseg_da': lseg_da,
            'lseg_ll': lseg_ll,
            'liou_ll': liou_ll
        }
        
        return loss, loss_dict


def get_loss(cfg, device):
    """
    Get the loss function for training
    """
    # BCE loss functions
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    
    # Create multi-head loss
    loss_list = [BCEcls, BCEobj, BCEseg]
    
    # Loss weights: [cls, obj, box, da_seg, ll_seg, ll_iou]
    lambdas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    criterion = MultiHeadLoss(loss_list, cfg, lambdas)
    
    return criterion