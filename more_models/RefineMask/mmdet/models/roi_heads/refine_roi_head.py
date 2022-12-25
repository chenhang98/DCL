import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2roi
from mmdet.models.losses.cross_entropy_loss import generate_block_target
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.builder import HEADS


@HEADS.register_module()
class RefineRoIHead(StandardRoIHead):
    def init_weights(self, pretrained):
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        # assign gts and sample proposals
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)

        mask_results = self._mask_forward_train(
            x, sampling_results, bbox_results['bbox_feats'], gt_bboxes, gt_masks, gt_labels, img_metas)

        losses = {}
        losses.update(bbox_results['loss_bbox'])
        losses.update(mask_results['loss_mask'])
        losses.update(mask_results['loss_semantic'])
        return losses

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_bboxes, gt_masks, gt_labels, img_metas):
        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        stage_mask_targets, semantic_target = \
            self.mask_head.get_targets(pos_bboxes, pos_assigned_gt_inds, gt_masks)

        mask_results = self._mask_forward(x, pos_rois, torch.cat(pos_labels))

        # resize the semantic target
        semantic_target = F.interpolate(
            semantic_target.unsqueeze(1),
            mask_results['semantic_pred'].shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        semantic_target = (semantic_target >= 0.5).float()

        loss_mask, loss_semantic = self.mask_head.loss(
            mask_results['stage_instance_preds'],
            mask_results['semantic_pred'],
            stage_mask_targets,
            semantic_target)

        mask_results.update(loss_mask=loss_mask, loss_semantic=loss_semantic)
        return mask_results

    def _mask_forward(self, x, rois, roi_labels):
        """Mask head forward function used in both training and testing."""

        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_instance_preds, semantic_pred = self.mask_head(ins_feats, x[0], rois, roi_labels)
        return dict(stage_instance_preds=stage_instance_preds, semantic_pred=semantic_pred)

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            mask_rois = bbox2roi([_bboxes])

            interval = 100  # to avoid memory overflow
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
            for i in range(0, det_labels.shape[0], interval):
                mask_results = self._mask_forward(x, mask_rois[i: i + interval], det_labels[i: i + interval])

                # refine instance masks from stage 1
                stage_instance_preds = mask_results['stage_instance_preds'][1:]
                for idx in range(len(stage_instance_preds) - 1):
                    instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
                    non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
                    non_boundary_mask = F.interpolate(
                        non_boundary_mask.float(),
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
                    pre_pred = F.interpolate(
                        stage_instance_preds[idx],
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                    stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
                instance_pred = stage_instance_preds[-1]

                chunk_segm_result = self.mask_head.get_seg_masks(
                    instance_pred, _bboxes[i: i + interval], det_labels[i: i + interval],
                    self.test_cfg, ori_shape, scale_factor, rescale)

                for c, segm in zip(det_labels[i: i + interval], chunk_segm_result):
                    segm_result[c].append(segm)

        return segm_result



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
                loss_weight: float=1.0,
                max_instances: int=100,
                num_samples: int=10, 
                temperature: float=0.1, 
                with_proj: bool=False,
                conv_dim: int=256,
                loss_func: str="contrast_loss"):
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

    def forward(self, feats: torch.Tensor, gt_masks: torch.Tensor, logits: torch.Tensor=None):
        # feats: N,C,H,W,   gt_masks: N,H,W
        if self.with_proj:
            feats = self.proj_head(feats)
        feats = F.normalize(feats, dim=1)       # N, C, H, W

        # sum over RoIs
        gt_masks = gt_masks.bool()
        num_instances = gt_masks.size(0)
        feats_side_len = gt_masks.size(1)
        losses = []
        num_instances = min(num_instances, self.max_instances)

        for i in range(num_instances):
            area = gt_masks[i].sum()
            if self.num_samples < area < feats_side_len**2 - self.num_samples:
                _loss = self.loss(feats[i], None, gt_masks[i], 
                                self.num_samples, tau=self.temperature)
                losses.append(_loss)

        if not len(losses):
            return feats.sum() * 0
        losses = torch.stack(losses)
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



@HEADS.register_module()
class RefineRoIHeadDCL(RefineRoIHead):
    def __init__(self,
                bbox_roi_extractor=None,
                bbox_head=None,
                mask_roi_extractor=None,
                mask_head=None,
                shared_head=None,
                train_cfg=None,
                test_cfg=None,
                contrast_cfg=None):
        super(RefineRoIHeadDCL, self).__init__(
                bbox_roi_extractor,
                bbox_head,
                mask_roi_extractor,
                mask_head,
                shared_head,
                train_cfg,
                test_cfg)
        self.contrast_loss = DenseContrastLoss(**contrast_cfg)

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        # assign gts and sample proposals
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)

        mask_results = self._mask_forward_train(
            x, sampling_results, bbox_results['bbox_feats'], gt_bboxes, gt_masks, gt_labels, img_metas)

        losses = {}
        losses.update(bbox_results['loss_bbox'])
        losses.update(mask_results['loss_mask'])
        losses.update(mask_results['loss_semantic'])
        losses.update(mask_results['loss_contrast'])
        return losses


    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_bboxes, gt_masks, gt_labels, img_metas):
        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        stage_mask_targets, semantic_target = \
            self.mask_head.get_targets(pos_bboxes, pos_assigned_gt_inds, gt_masks)

        mask_results = self._mask_forward(x, pos_rois, torch.cat(pos_labels))

        # contrast loss here
        loss_contrast = self.contrast_loss(mask_results['instance_feats'], stage_mask_targets[1])
        loss_contrast = {'loss_contrast': loss_contrast}

        # resize the semantic target
        semantic_target = F.interpolate(
            semantic_target.unsqueeze(1),
            mask_results['semantic_pred'].shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        semantic_target = (semantic_target >= 0.5).float()

        loss_mask, loss_semantic = self.mask_head.loss(
            mask_results['stage_instance_preds'],
            mask_results['semantic_pred'],
            stage_mask_targets,
            semantic_target)

        mask_results.update(loss_mask=loss_mask, loss_semantic=loss_semantic, loss_contrast=loss_contrast)
        return mask_results


    def _mask_forward(self, x, rois, roi_labels):
        """Mask head forward function used in both training and testing."""

        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_instance_preds, semantic_pred, instance_feats = \
                    self.mask_head(ins_feats, x[0], rois, roi_labels, return_instance_feats=True)
        return dict(stage_instance_preds=stage_instance_preds, 
                    semantic_pred=semantic_pred,
                    instance_feats=instance_feats)



@HEADS.register_module()
class SimpleRefineRoIHead(StandardRoIHead):

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in training."""

        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        mask_results = self._mask_forward(x, pos_rois, torch.cat(pos_labels))
        stage_mask_targets = self.mask_head.get_targets(pos_bboxes, pos_assigned_gt_inds, gt_masks)
        loss_mask = self.mask_head.loss(mask_results['stage_instance_preds'], stage_mask_targets)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def _mask_forward(self, x, rois, roi_labels):
        """Mask head forward function used in both training and testing."""

        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_instance_preds = self.mask_head(ins_feats, x[0], rois, roi_labels)
        return dict(stage_instance_preds=stage_instance_preds)

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            mask_rois = bbox2roi([_bboxes])

            interval = 100  # to avoid memory overflow
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
            for i in range(0, det_labels.shape[0], interval):
                mask_results = self._mask_forward(x, mask_rois[i: i + interval], det_labels[i: i + interval])

                # refine instance masks from stage 1
                stage_instance_preds = mask_results['stage_instance_preds'][1:]
                for idx in range(len(stage_instance_preds) - 1):
                    instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
                    non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
                    non_boundary_mask = F.interpolate(
                        non_boundary_mask.float(),
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
                    pre_pred = F.interpolate(
                        stage_instance_preds[idx],
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                    stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
                instance_pred = stage_instance_preds[-1]

                chunk_segm_result = self.mask_head.get_seg_masks(
                    instance_pred, _bboxes[i: i + interval], det_labels[i: i + interval],
                    self.test_cfg, ori_shape, scale_factor, rescale)

                for c, segm in zip(det_labels[i: i + interval], chunk_segm_result):
                    segm_result[c].append(segm)

        return segm_result
