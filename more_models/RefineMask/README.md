# RefineMask + DCL

We apply DCL by modifying the official code of [RefineMask](https://github.com/zhanggang001/RefineMask) which is based on [mmdetection](https://github.com/open-mmlab/mmdetection) instead of detectron2.
Since few changes are required, we only release the modified parts.

One can apply DCL to other models of the mmdetection library with similar modifications. 


## Installation


* First, install RefineMask following its [official repository](https://github.com/zhanggang001/RefineMask). Note that our code was tested on commit `633ed2b`.

* Replace the corresponding files (i.e. `mmdet/models/roi_heads/refine_roi_head.py` etc).


## Training

Run the following commands to train RefineMask + DCL on Cityscapes dataset. Note that we use 1.4 as the loss weight for DCL, which generates slightly better results than the paper (1.0).

```
./scripts/dist_train.sh \
    configs/refinemask/cityscapes/r50-refinemask-dcl-w14.py\
    8 \
    work_dirs/r50-refinemask-dcl-w14-cityscapes
```


## Evaluation

* Generate Cityscapes-style results on validation set:

    ```bash
    ./scripts/dist_test.sh \
        configs/refinemask/cityscapes/r50-refinemask-dcl-w14.py \
        r50-refinemask-dcl-w14-cityscapes.pth \
        8  \
        --format-only \
        --options "txtfile_prefix=./r50-refinemask-dcl-w14-val"
    ```

* Evaluate the results:

    ```bash
    CITYSCAPES_DATASET=data/cityscapes/ \
    CITYSCAPES_RESULTS=r50-refinemask-dcl-w14-val \
    python -m cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling
    ```