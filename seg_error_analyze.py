from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import copy
import json
import os 
import os.path as osp
import numpy as np
from tqdm import tqdm
import argparse
from tabulate import tabulate
from termcolor import colored
import multiprocessing as mp
from functools import partial

from detectron2.evaluation.fast_eval_api import COCOeval_opt


def coco_eval_d2(ann_file, res_file, iou_type="segm", 
            header=["AP", "AP50", "AP75", "APs", "APm", "APl", 
                "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]):
    coco_results = json.load(open(res_file))
    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval_opt(coco_gt, coco_dt, iou_type)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats, header


def xywh2xyxy(boxes):
    boxes = boxes.copy()
    boxes[:,2] += boxes[:,0]   # xywh -> xyxy
    boxes[:,3] += boxes[:,1]
    return boxes


def pairwise_box_iou(boxes1, boxes2):
    # refer detectron2
    # boxes1, boxes2: shape [N,4], format x1, y1, x2, y2
    width_height = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:]) \
        - np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    width_height = width_height.clip(min=0)   # [N,M,2]
    inter = width_height.prod(axis=2)  # [N,M]
    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    iou = np.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        0
    )
    return iou


def pairwise_mask_iou(masks1, masks2):
    # masks1, masks2: shape [N,H,W], bool type
    inter = (masks1[:, None] & masks2).sum(-1).sum(-1)
    inter = inter.astype(float)
    area1 = masks1.sum(-1).sum(-1)
    area2 = masks2.sum(-1).sum(-1)
    iou = np.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        0
    )
    return iou


def anns2boxes(anns, format='xyxy'):
    assert format in ['xyxy', 'xxyy']
    boxes = np.array([_['bbox'] for _ in anns])
    if format == 'xyxy':
         return xywh2xyxy(boxes)
    return boxes


def annToMask(ann, h, w):
    # rle = self.annToRLE(ann)
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']

    m = maskUtils.decode(rle)
    return m


def anns2masks(anns, h, w):
    # masks = [COCO.annToMask(COCO, _) for _ in anns]
    masks = [annToMask(_, h, w) for _ in anns]
    masks = np.stack(masks).astype(bool)
    return masks


def class_aware_iou(dts, gts, iou_type, h, w):
    assert iou_type in {'seg', 'box'}

    G = len(gts)
    D = len(dts)

    ious = np.zeros((D, G))
    dts.sort(key=lambda x: x['category_id'])
    gts.sort(key=lambda x: x['category_id'])

    dt_masks = anns2masks(dts, h, w)
    gt_masks = anns2masks(gts, h, w)

    if iou_type == 'box':
        dt_boxes = anns2boxes(dts)
        gt_boxes = anns2boxes(gts)

    cats_dt = np.array([d['category_id'] for d in dts])
    cats_gt = np.array([g['category_id'] for g in gts])

    cs = set(cats_dt.tolist()) & set(cats_gt.tolist())
    for c in cs:
        ind_dt = cats_dt==c
        ind_gt = cats_gt==c
        ind = ind_dt[:,None] & ind_gt[None,:]

        if iou_type == 'box':
            ious[ind] = pairwise_box_iou(dt_boxes[ind_dt], gt_boxes[ind_gt]).flatten()
        else:
            ious[ind] = pairwise_mask_iou(dt_masks[ind_dt], gt_masks[ind_gt]).flatten()

    return ious, dts, gts, dt_masks, gt_masks


def matching(dts, gts, ious, t=0.1):
    # matching each gt with multiple gts
    G, D = len(gts), len(dts)
    dtm = -1 * np.ones(D, dtype=int)
    for dind, d in enumerate(dts):
        # information about best match so far (m=-1 -> unmatched)
        iou = min([t, 1-1e-10])
        m   = -1
        for gind, g in enumerate(gts):
            # continue to next gt unless better match made
            if ious[dind,gind] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou=ious[dind,gind]
            m=gind
        # if match made store id of match for both dt and gt
        if m ==-1:
            continue
        dtm[dind]  = m
    return dtm


def masks2anns(masks, dts):
    assert len(masks) == len(dts)
    dts = copy.deepcopy(dts)
    for mask, dt in zip(masks, dts):
        dt['segmentation'] = maskUtils.encode(mask)
        dt['segmentation']['counts'] = dt['segmentation']['counts'].decode('utf8')
    return dts


def remove_fFP(dts, gts, dt_masks, gt_masks, dtm):
    # remove foreground FP
    fg_mask = gt_masks.any(0)
    other_masks = []
    for i, mask in enumerate(gt_masks):
        other_masks.append(fg_mask & ~mask)
    other_masks = np.stack(other_masks)
    processed_dt_masks = dt_masks & ~other_masks[dtm]
    processed_dts = masks2anns(processed_dt_masks, dts)
    return processed_dts, processed_dt_masks


def remove_bFP(dts, gts, dt_masks, gt_masks, dtm):
    # remove background FP
    fg_mask = gt_masks.any(0)
    processed_dt_masks = dt_masks & fg_mask
    processed_dts = masks2anns(processed_dt_masks, dts)
    return processed_dts, processed_dt_masks


def remove_FN(dts, gts, dt_masks, gt_masks, dtm):
    # remove FN
    gt_masks = gt_masks[dtm]
    processed_dt_masks = dt_masks | gt_masks
    processed_dts = masks2anns(processed_dt_masks, dts)
    return processed_dts, processed_dt_masks


def combine(*ops):
    def new_op(dts, gts, dt_masks, gt_masks, dtm):
        for op in ops:
            dts, dt_masks = op(dts, gts, dt_masks, gt_masks, dtm)
        return dts, dt_masks
    return new_op


def parse_types(s):
    if not '&' in s:
        return f'remove_{s}'
    else:
        return 'combine(' + ', '.join(parse_types(t) for t in s.split('&')) + ')'


def make_or_ignore(path):
    if not osp.exists(path):
        os.makedirs(path)


def run_image(arg, code, iou_type):
    img_id, gts, dts, h, w = arg
    if (not len(gts)) or (not len(dts)):
        return dts

    ious, dts, gts, dt_masks, gt_masks = class_aware_iou(dts, gts, iou_type, h, w)
    dtm = matching(dts, gts, ious, t=0.1)

    processed_dts, processed_dt_masks = eval(code)(dts, gts, dt_masks, gt_masks, dtm)
    return processed_dts


def filter_crowd(gts):
    return [_ for _ in gts if not _['iscrowd']]


def benchmark(res_file, ann_file, temp_dir, error_types, iou_type, num_processes=mp.cpu_count()//2):
    coco_gt = COCO(ann_file)
    make_or_ignore(temp_dir)

    summary_table = []
    for t in error_types:
        code = parse_types(t)
        coco_dt = coco_gt.loadRes(res_file)

        out_file = osp.basename(res_file).split('.')[0] + f'_{code}.json'
        out_file = osp.join(temp_dir, out_file)

        print(colored(f"Executing {code}, file saved to {out_file}", "green"))

        res = []
        img_ids = list(coco_dt.imgs.keys())
        with mp.Pool(processes=num_processes) as p:
            with tqdm(total=len(img_ids)) as pbar:

                print('Preparing args...')
                DTs = [coco_dt.imgToAnns[img_id] for img_id in img_ids]
                GTs = [filter_crowd(coco_gt.imgToAnns[img_id]) for img_id in img_ids]
                Hs  = [coco_gt.imgs[img_id]['height'] for img_id in img_ids]
                Ws  = [coco_gt.imgs[img_id]['width']  for img_id in img_ids]
                args = list(zip(img_ids, GTs, DTs, Hs, Ws))
                print("done")

                _func = partial(run_image, code=code, iou_type=iou_type)
                for dts in p.imap_unordered(_func, args):
                    res.extend(dts)
                    pbar.update()

        with open(out_file, 'w') as f:
            json.dump(res, f)

        stats_before, headers = coco_eval_d2(ann_file, res_file, iou_type="segm")
        stats_after, _ = coco_eval_d2(ann_file, out_file, iou_type="segm")
        stats_before *= 100
        stats_after *= 100
        delta_stats = stats_after - stats_before

        table = [
            ['baseline', *stats_before],
            [f'+{code}', *stats_after],
            ['delta', *delta_stats]
        ]
        print(tabulate(table, ['', *headers], tablefmt="pipe", floatfmt=".2f"))
        if not len(summary_table):
            summary_table.append(table[0])
        summary_table.append([f'{t}', *delta_stats])

    print(colored(f"Summraize {', '.join(error_types)}", "green"))
    if len(summary_table):
        print(tabulate(summary_table, ['', *headers], tablefmt="pipe", floatfmt=".2f"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="An analysis tool to evaluate the severity of different types of segmentation errors in instance segmentation."
    )
    parser.add_argument('res_file', type=str,
                    help='the prediction json file.')
    parser.add_argument('--types', '-t', type=str, nargs='+', default=['fFP', 'bFP', 'FN'],
                    help='the type of error to remove. Using \"&\" to remove multiple errors at the same time, e.g. fFP&bFP.')
    parser.add_argument('--ann_file', '-a', type=str, default='datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
                    help='the annotation file of the dataset.')
    parser.add_argument('--temp_dir', '-d', type=str, default='temp',
                    help='the directory to save the intermediate json results.')
    parser.add_argument('--iou_type', '-i', type=str, default='seg', choices=['box', 'seg'],
                    help='the type of iou used for matching.')
    parser.add_argument('--num_processes', '-n', type=int, default=mp.cpu_count()//2,
                    help='the number of processes.')
    args = parser.parse_args()

    benchmark(args.res_file, 
            args.ann_file, 
            args.temp_dir, 
            args.types, 
            args.iou_type, 
            args.num_processes)
