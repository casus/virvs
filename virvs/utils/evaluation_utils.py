from collections import defaultdict
from math import floor, log10

import numpy as np
from skimage.metrics import structural_similarity

from virvs.utils.metrics_utils import calculate_metrics


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def calculate_acc(mask1, mask2):
    return np.mean(mask1 == mask2)


def calculate_acc_only_cells(mask_gt, mask_pred, background_px):
    return (np.sum(mask_gt == mask_pred) - background_px) / (
        mask_gt.size - background_px
    )


def calculate_prec(mask_gt, mask_pred, background_px):
    false_negatives, false_positives, true_negatives, true_positives = get_stats(
        mask_gt, mask_pred, background_px
    )
    return true_positives / (true_positives + false_positives + 1e-6)


def calculate_cell_precision(masks_pred, mask_gt, new_pred_mask):
    fp_mask = ((masks_pred * new_pred_mask) != 0) & ((masks_pred * mask_gt) == 0)
    fp_cellcount = np.unique(masks_pred[fp_mask]).size

    tp_mask = ((masks_pred * new_pred_mask) != 0) & ((masks_pred * mask_gt) != 0)
    tp_cellcount = np.unique(masks_pred[tp_mask]).size
    return tp_cellcount / (tp_cellcount + fp_cellcount + 1e-6)


def calculate_rec(mask_gt, mask_pred, background_px):
    false_negatives, false_positives, true_negatives, true_positives = get_stats(
        mask_gt, mask_pred, background_px
    )
    return true_positives / (true_positives + false_negatives + 1e-6)


def calculate_cell_rec(masks_pred, mask_gt, new_pred_mask):
    fn_mask = ((masks_pred * new_pred_mask) == 0) & ((masks_pred * mask_gt) != 0)
    fn_cellcount = np.unique(masks_pred[fn_mask]).size

    tp_mask = ((masks_pred * new_pred_mask) != 0) & ((masks_pred * mask_gt) != 0)
    tp_cellcount = np.unique(masks_pred[tp_mask]).size
    return tp_cellcount / (tp_cellcount + fn_cellcount + 1e-6)


def get_stats(mask_gt, mask_pred, background_px):

    true_positives = np.sum(np.logical_and(mask_gt, mask_pred))
    true_negatives = np.sum(np.logical_and(~mask_gt, ~mask_pred)) - background_px
    false_positives = np.sum(np.logical_and(~mask_gt, mask_pred))
    false_negatives = np.sum(np.logical_and(mask_gt, ~mask_pred))

    return false_negatives, false_positives, true_negatives, true_positives


def calculate_f1(mask_gt, mask_pred, background_px):
    fn, fp, tn, tp = get_stats(mask_gt, mask_pred, background_px)
    return 2 * tp / (2 * tp + fp + fn)


def get_masks_num_and_area(masks_pred, new_mask):
    unique_highlighted_masks = np.unique(masks_pred[new_mask])
    mask_areas = {mask: np.sum(masks_pred == mask) for mask in unique_highlighted_masks}
    return (len(unique_highlighted_masks), sum(mask_areas.values()))


def get_masks_to_show(mean_per_mask, threshold):
    masks_to_show = np.where(mean_per_mask > threshold)[0]
    if 0 in masks_to_show:
        index = np.argwhere(masks_to_show == 0)
        masks_to_show = np.delete(masks_to_show, index)
    return masks_to_show


def get_mean_per_mask(mask_flat, weights):
    max_mask_value = mask_flat.max()
    sum_per_mask = np.bincount(mask_flat, weights=weights, minlength=max_mask_value + 1)
    count_per_mask = np.bincount(mask_flat, minlength=max_mask_value + 1)
    mean_per_mask = sum_per_mask / count_per_mask
    return mean_per_mask


def ssim_psnr(pred, label):
    ssim = structural_similarity(
        np.squeeze(pred),
        np.squeeze(label),
        data_range=2,
    )
    mse = np.mean((pred - label) ** 2)
    psnr = 10 * np.log10((2**2) / mse)
    return ssim, psnr


def evaluate(preds, gts, masks=None):
    cumulative_metrics = defaultdict(list)
    for pred, gt, mask in zip(preds, gts, masks):
        metrics = calculate_metrics(pred, gt)
        if masks is not None:
            fg_ssim, fg_psnr = ssim_psnr(pred[mask != 0], gt[mask != 0])
            bg_ssim, bg_psnr = ssim_psnr(pred[mask == 0], gt[mask == 0])
        for k, v in metrics.items():
            cumulative_metrics[k].append(v)
        if masks is not None:
            cumulative_metrics["fg_ssim"].append(fg_ssim)
            cumulative_metrics["fg_psnr"].append(fg_psnr)
            cumulative_metrics["bg_ssim"].append(bg_ssim)
            cumulative_metrics["bg_psnr"].append(bg_psnr)
    for k, v in cumulative_metrics.items():
        print(k, np.mean(np.array(v)))


def round_to_1(x):
    position = -int(floor(log10(abs(x))))
    if position < 3:
        position = 3
    return round(x, position)
