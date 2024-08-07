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

    mean_metrics = {key: np.mean(values) for key, values in test_metrics.items()}

    return mean_metrics


BASE_PATH = "/home/wyrzyk93/VIRVS/outputs/weights/"
WEIGHTS = {
    "pix2pix": {
        "hadv_2ch": "model_200000_38527895-6967-4f7b-a5df-eb6564b47ad9.h5",
        "hadv_1ch": "model_200000_63065c3c-b6bd-4a38-a5b9-9d9ae5083129.h5",
    },
    "unet": {
        "hadv_2ch": "model_200000_633f9e8a-0c3b-4678-9a67-ec351fdf09de.h5",
        "hadv_1ch": "model_200000_b301ae59-b3a9-4faf-9ae1-e3b805712147.h5",
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
        if model == "unet":
            tf.keras.utils.set_random_seed(42)
            generator = Generator(size, ch_in=ch_in, ch_out=1, apply_dropout=dropout)
            generator.load_weights(f"{BASE_PATH}/{WEIGHTS[model][virus]}")
            test_metrics = eval(dataloader, generator, masks)
            for key, value in test_metrics.items():
                print(f"{key}: {round(value, 3)}")
        else:
            seeds = [42, 43, 44]
            all_seeds_metrics = defaultdict(list)
            generator = Generator(size, ch_in=ch_in, ch_out=1, apply_dropout=dropout)
            generator.load_weights(f"{BASE_PATH}/{WEIGHTS[model][virus]}")
            for seed in seeds:
                tf.keras.utils.set_random_seed(seed)
                test_metrics = eval(dataloader, generator, masks)
                for k, v in test_metrics.items():
                    all_seeds_metrics[k].append(np.mean(v))

            results = {}
            for k, v in all_seeds_metrics.items():
                mean_metric = np.mean(v)
                std_metric = np.std(v)
                results[k] = {"mean": mean_metric, "std": std_metric}

            for key, value in results.items():
                print(
                    f"{key}: mean = {round(value['mean'], 3)}, std = {round_to_1(value['std']):.1e}"
                )
