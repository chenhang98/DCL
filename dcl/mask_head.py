import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.structures import ImageList, Instances
from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.layers import ShapeSpec, cat 
from detectron2.modeling.roi_heads.mask_head import \
    MaskRCNNConvUpsampleHead, mask_rcnn_inference, mask_rcnn_loss, \
    ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY, \
    select_foreground_proposals


def get_gt_masks(instances, side_len, device):
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, side_len
        ).to(device=device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks):
        return cat(gt_masks, dim=0)
    return gt_masks


def _sample(mask, num_samples):
    inds = torch.nonzero(mask)
    return inds[torch.randint(inds.size(0), size=(num_samples,))]


def auto_sample(func):
    def func_with_sample(feats, logits, gt_mask, num_samples, **kw):
        anchor_inds = _sample(gt_mask, num_samples)
        pos_inds    = _sample(gt_mask, num_samples)
        neg_inds    = _sample(~gt_mask, num_samples)
        return func(feats, logits, gt_mask, anchor_inds, pos_inds, neg_inds, **kw)
    return func_with_sample


class DenseContrastLoss(nn.Module):
    def __init__(self, 
                loss_weight: float=1.2,
                max_instances: int=100,
                num_samples: int=32,
                temperature: float=0.07,
                with_proj: bool=True,
                conv_dim: int=256,
                loss_func: str="dense_contrast_loss"):
        super().__init__()
        self.loss_weight = loss_weight
        self.max_instances = max_instances
        self.num_samples = num_samples
        self.temperature = temperature
        self.with_proj = with_proj
        if self.with_proj:
            self.proj_head = nn.Sequential(
                nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=1),
            )
        self.loss = getattr(self, loss_func)


    def forward(self, feats: torch.Tensor, instances: List[Instances], 
                targets: List[Instances], logits: torch.Tensor=None):
        if self.with_proj:
            feats = self.proj_head(feats)
        feats = F.normalize(feats, dim=1)       # N, C, H, W

        # GT masks
        num_instances, _, feats_side_len, _ = feats.shape
        gt_masks = get_gt_masks(instances, feats_side_len, feats.device)
        if len(gt_masks) == 0:
            return feats.sum() * 0
        assert gt_masks.dtype == torch.bool

        gt_classes = cat([_.gt_classes for _ in instances], dim=0)

        # sum over RoIs
        losses = []
        num_instances = min(num_instances, self.max_instances)

        for i in range(num_instances):
            area = gt_masks[i].sum()
            if self.num_samples < area < feats_side_len**2 - self.num_samples:
                _loss = self.loss(feats[i], logits[i, gt_classes[i]], gt_masks[i], 
                                self.num_samples, tau=self.temperature)
                losses.append(_loss)

        storage = get_event_storage()
        storage.put_scalar("contrast_loss/num_instances", num_instances)
        storage.put_scalar("contrast_loss/valid_instances", len(losses))

        if not len(losses):
            return feats.sum() * 0
        losses = torch.stack(losses)
        storage.put_histogram("contrast_loss/distribution", losses)
        return losses.mean() * self.loss_weight

    @staticmethod
    @auto_sample
    def dense_contrast_loss(feats, logits, gt_mask, anchor_inds, pos_inds, neg_inds, tau=0.07):
        # feats: tensor, shape C, H, W
        # *_inds: tensor, shape K, 2
        anchor_feats = feats[:, anchor_inds[:,0], anchor_inds[:,1]].T    # K, C
        pos_feats = feats[:, pos_inds[:,0], pos_inds[:,1]].T
        neg_feats = feats[:, neg_inds[:,0], neg_inds[:,1]].T

        # *_feats: tensor, shape N, C
        sim_anchor_pos = torch.matmul(anchor_feats, pos_feats.T)    # A, P
        sim_anchor_pos /= tau
        sim_anchor_neg = torch.matmul(anchor_feats, neg_feats.T)    # A, N
        sim_anchor_neg /= tau
        
        sim_max = sim_anchor_pos.max(1, keepdim=True)[0].detach()   # A, 1
        sim_anchor_pos -= sim_max
        sim_anchor_neg -= sim_max
        
        exp_anchor_neg = sim_anchor_neg.exp().sum(1, keepdim=True)  # A, 1
        exp_anchor_pos = sim_anchor_pos.exp()                       # A, P
        
        loss_anchor_pos = -torch.log(exp_anchor_pos / (exp_anchor_pos + exp_anchor_neg))    # A, P
        loss = loss_anchor_pos.mean()
        
        return loss



@ROI_MASK_HEAD_REGISTRY.register()
class ContrastConvUpsampleHead(MaskRCNNConvUpsampleHead):

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", contrast_loss_cfg=None, **kwargs):
        super(ContrastConvUpsampleHead, self).__init__(input_shape, 
                    num_classes=num_classes, 
                    conv_dims=conv_dims, 
                    conv_norm=conv_norm, 
                    **kwargs)
        if contrast_loss_cfg is not None:
            self.contrast_loss = DenseContrastLoss(**contrast_loss_cfg, conv_dim=conv_dims[-1])

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        _cfg = cfg.MODEL.ROI_MASK_HEAD.CONTRAST_LOSS
        ret["contrast_loss_cfg"] = dict(
            loss_weight = _cfg.LOSS_WEIGHT,
            max_instances = _cfg.MAX_INSTANCES,
            num_samples = _cfg.NUM_SAMPLES,
            temperature = _cfg.TEMPERATURE,
            with_proj = _cfg.WITH_PROJ,
            loss_func = _cfg.LOSS_FUNC,
        )
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, instances: List[Instances], targets: List[Instances]=None):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        feat = None
        layers = [_ for _ in self if not isinstance(_, DenseContrastLoss)]

        for i, layer in enumerate(layers):
            x = layer(x)
            if self.training and (i == len(layers)-2):
                feat = x

        if self.training:
            loss_contrast = self.contrast_loss(feat, instances, targets, logits=x)
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period) * self.loss_weight,
                    "loss_contrast": loss_contrast}
        else:
            mask_rcnn_inference(x, instances)
            return instances


@ROI_HEADS_REGISTRY.register()
class ContrastRoIHeads(StandardROIHeads):
    """ StandardROIHeads with little modification to 
    pass in the image-level ground truth to the Mask heads."""

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals, targets))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances], targets: List[Instances]=None):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances, targets)

