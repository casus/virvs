from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from virvs.architectures.pix2pix import Generator
from virvs.data.npy_dataloader import NpyDataloader
from virvs.utils.evaluation_utils import (
    calculate_acc,
    calculate_acc_only_cells,
    calculate_cell_precision,
    calculate_cell_rec,
    calculate_f1,
    calculate_iou,
    calculate_prec,
    calculate_rec,
    evaluate,
    get_masks_to_show,
    get_mean_per_mask,
    round_to_1,
)


def eval(dataloader, generator, masks):
    test_metrics = defaultdict(list)

    for i in tqdm(range(len(dataloader))):
        x, y = dataloader[i]
        masks_pred = masks[i]
        pred = np.squeeze(generator(np.expand_dims(x, 0), training=True), 0)

        label_flat = y.flatten()
        mask_flat = masks_pred.flatten()

        weights = (label_flat > -0.9).astype(np.float32)
        mean_per_mask = get_mean_per_mask(mask_flat, weights)
        masks_to_show = get_masks_to_show(mean_per_mask, 0.5)
        new_mask = np.isin(masks_pred, masks_to_show)

        pred_weights = (np.squeeze(pred).flatten() > -0.9).astype(np.float32)
        pred_mean_per_mask = get_mean_per_mask(mask_flat, pred_weights)
        pred_masks_to_show = get_masks_to_show(pred_mean_per_mask, 0.5)
        pred_new_mask = np.isin(masks_pred, pred_masks_to_show)
        if np.all(new_mask == 0):
            continue
        test_metrics["iou"].append(calculate_iou(pred_new_mask, new_mask))
        test_metrics["f1"].append(
            calculate_f1(new_mask, pred_new_mask, np.sum(masks_pred == 0))
        )
        test_metrics["acc"].append(calculate_acc(pred_new_mask, new_mask))
        test_metrics["acc_only_cells"].append(
            calculate_acc_only_cells(new_mask, pred_new_mask, np.sum(masks_pred == 0))
        )
        test_metrics["prec"].append(
            calculate_prec(new_mask, pred_new_mask, np.sum(masks_pred == 0))
        )
        test_metrics["cell_prec"].append(
            calculate_cell_precision(masks_pred, new_mask, pred_new_mask)
        )
        test_metrics["rec"].append(
            calculate_rec(new_mask, pred_new_mask, np.sum(masks_pred == 0))
        )
        test_metrics["cell_rec"].append(
            calculate_cell_rec(masks_pred, new_mask, pred_new_mask)
        )

    return test_metrics


BASE_PATH = "/home/wyrzyk93/VIRVS/outputs/weights/"
WEIGHTS = {
    "pix2pix": {
        "hadv_2ch": "model_100000_316674a2-4299-4a06-b601-5e20f7dd02a6.h5",
        "hadv_1ch": "model_100000_d6792b38-8091-448d-a26a-ef08375b8dbe.h5",
    },
    "unet": {
        "hadv_2ch": "model_100000_c0175f01-1e6e-4bec-9f2e-2e5542c16584.h5",
        "hadv_1ch": "model_100000_47b9346c-e143-4bf7-9dce-40aa6f2329e5.h5",
    },
}

masks = np.load(f"/bigdata/casus/MLID/maria/VIRVS_data/masks/masks_cell_hadv_test.npy")

for virus in ["hadv_2ch", "hadv_1ch"]:

    size = 2048
    if "2ch" in virus:
        ch_in = [0, 1]
    else:
        ch_in = [0]

    dataloader = NpyDataloader(
        path="/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
        im_size=2048,
        random_jitter=False,
        ch_in=ch_in,
        crop_type="center",
    )
    for model in ["unet", "pix2pix"]:
        if model == "unet":
            dropout = False
        else:
            dropout = True
        print(model, ", ", virus)
        tf.keras.utils.set_random_seed(42)
        generator = Generator(size, ch_in=ch_in, ch_out=1, apply_dropout=dropout)
        generator.load_weights(f"{BASE_PATH}/{WEIGHTS[model][virus]}")
        test_metrics = eval(dataloader, generator, masks)
        for key, value in test_metrics.items():
            print(f"{key}: {round(np.mean(value), 3)}, std {round(np.std(value), 3)}")