# Cityscapes
CUDA_VISIBLE_DEVICES=5,6,7,8 \
python train_net.py \
    --num-gpus 4 \
    --config-file configs/cityscapes_mask_rcnn_R_50_FPN.yaml \
    SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ../detectron2/detectron2_ImageNetPretrained_MSRA_R-50.pkl \
    MODEL.ROI_HEADS.NAME ContrastRoIHeads \
    MODEL.ROI_MASK_HEAD.NAME ContrastConvUpsampleHead \
    OUTPUT_DIR work_dirs/cityscapes_mask_rcnn_dcl



CUDA_VISIBLE_DEVICES=0,9 \
python train_net.py \
    --eval-only \
    --num-gpus 2 \
    --config-file configs/cityscapes_mask_rcnn_R_50_FPN.yaml \
    SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS work_dirs/cityscapes_mask_rcnn_dcl/model_final.pth \
    OUTPUT_DIR work_dirs/cityscapes_mask_rcnn_dcl



# COCO
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python train_net.py \
    --num-gpus 4 \
    --config-file configs/coco_mask_rcnn_R_50_FPN_1x.yaml \
    SOLVER.IMS_PER_BATCH 16 \
    MODEL.ROI_MASK_HEAD.CONTRAST_LOSS.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.NAME ContrastRoIHeads \
    MODEL.ROI_MASK_HEAD.NAME ContrastConvUpsampleHead \
    OUTPUT_DIR work_dirs/coco_mask_rcnn_dcl


CUDA_VISIBLE_DEVICES=0,9 \
python train_net.py \
    --eval-only \
    --num-gpus 2 \
    --config-file configs/coco_mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.WEIGHTS work_dirs/coco_mask_rcnn_dcl/model_final.pth \
    OUTPUT_DIR work_dirs/coco_mask_rcnn_dcl
