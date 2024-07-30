from collections import defaultdict

import numpy as np
from tqdm import tqdm
from virvs.data.npy_dataloader import NpyDataloader
from virvs.utils.evaluation_utils import (
    calculate_acc,
    calculate_acc_only_cells,
    calculate_cell_precision,
    calculate_f1,
    calculate_iou,
    calculate_prec,
    evaluate,
    get_masks_to_show,
    get_mean_per_mask,
)

outputs = np.load(f"")
cellpose_outputs = np.load(f"")

dataloader = NpyDataloader(
    path="",
    im_size=2048,
    random_jitter=False,
    ch_in=[0, 1],
)

evaluate(outputs, dataloader._y, cellpose_outputs)

iou = []
f1 = []
acc = []
acc_only_cells = []
prec = []
cell_prec = []

mean, std = np.mean(outputs), np.std(outputs)
gt_mean, gt_std = np.mean(dataloader._y), np.std(dataloader._y)

for i in tqdm(range(len(dataloader))):
    x, y, _ = dataloader[i]
    masks_pred = cellpose_outputs[i]
    pred = outputs[i]

    label_flat = y.flatten()
    mask_flat = masks_pred.flatten()

    mean_per_mask = get_mean_per_mask(mask_flat, label_flat)
    masks_to_show = get_masks_to_show(mean_per_mask, gt_mean + 0.5 * gt_std)

    new_mask = np.isin(masks_pred, masks_to_show)
    highlighted_image = np.where(new_mask, x[..., 0], -1)

    pred_mean_per_mask = get_mean_per_mask(mask_flat, np.squeeze(pred).flatten())
    pred_masks_to_show = get_masks_to_show(pred_mean_per_mask, mean + 0.5 * std)

    pred_new_mask = np.isin(masks_pred, pred_masks_to_show)

    iou.append(calculate_iou(pred_new_mask, new_mask))
    f1.append(calculate_f1(new_mask, pred_new_mask, np.sum(masks_pred == 0)))
    acc.append(calculate_acc(pred_new_mask, new_mask))
    acc_only_cells.append(
        calculate_acc_only_cells(new_mask, pred_new_mask, np.sum(masks_pred == 0))
    )
    prec.append(calculate_prec(new_mask, pred_new_mask, np.sum(masks_pred == 0)))
    cell_prec.append(calculate_cell_precision(masks_pred, new_mask, pred_new_mask))

iou = np.array(iou)
print(np.mean(iou))
f1 = np.array(f1)
print(np.mean(f1))
acc = np.array(acc)
print(np.mean(acc))
acc_only_cells = np.array(acc_only_cells)
print(np.mean(acc_only_cells))
prec = np.array(prec)
print(np.mean(prec))
cell_prec = np.array(cell_prec)
print(np.mean(cell_prec))
