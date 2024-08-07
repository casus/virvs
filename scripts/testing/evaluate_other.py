from collections import defaultdict
from math import floor, log10

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from virvs.architectures.pix2pix import Generator
from virvs.data.npy_dataloader import NpyDataloader, center_crop
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


def eval(dataloader, generator, masks, threshold):
    test_metrics = defaultdict(list)

    for i in tqdm(range(len(dataloader))):
        x, y = dataloader[i]
        masks_pred = center_crop(masks[i], x.shape[0])
        pred = np.squeeze(generator(np.expand_dims(x, 0), training=True), 0)

        label_flat = y.flatten()
        mask_flat = masks_pred.flatten()

        weights = (label_flat > threshold).astype(np.float32)
        mean_per_mask = get_mean_per_mask(mask_flat, weights)
        masks_to_show = get_masks_to_show(mean_per_mask, 0.5)
        new_mask = np.isin(masks_pred, masks_to_show)

        pred_weights = (np.squeeze(pred).flatten() > threshold).astype(np.float32)
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


DATASETS = {
    "hadv_2ch": "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
    "hadv_1ch": "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
    "vacv": "/bigdata/casus/MLID/maria/VIRVS_data/VACV/processed/test",
    "iav": "/bigdata/casus/MLID/maria/VIRVS_data/IAV/processed/test",
    "hsv": "/bigdata/casus/MLID/maria/VIRVS_data/HSV/processed/test",
    "rv": "/bigdata/casus/MLID/maria/VIRVS_data/RV/processed/test",
}

BASE_PATH = "/home/wyrzyk93/VIRVS/outputs/weights/"
WEIGHTS = {
    "pix2pix": {
        "hadv_2ch": "model_200000_38527895-6967-4f7b-a5df-eb6564b47ad9.h5",
        "hadv_1ch": "model_200000_63065c3c-b6bd-4a38-a5b9-9d9ae5083129.h5",
        "iav": "model_200000_fbc9c491-14a6-4ed4-a02d-daefd99fd60e.h5",
        "hsv": "model_200000_afb8f712-ae29-404e-bcd9-a06786e5952b.h5",
        "rv": "model_200000_964056e1-a1e3-4c4f-9e9e-21112f54453a.h5",
    },
    "unet": {
        "hadv_2ch": "model_200000_633f9e8a-0c3b-4678-9a67-ec351fdf09de.h5",
        "hadv_1ch": "model_200000_b301ae59-b3a9-4faf-9ae1-e3b805712147.h5",
        "iav": "model_200000_868408c5-7b76-412b-9a8d-faf4812ed96f.h5",
        "hsv": "model_200000_1d7dded0-481b-4a29-b3d5-16899f219b6e.h5",
        "rv": "model_200000_d88881a7-e0c3-455f-bb5d-521cfbb98329.h5",
    },
}


for virus in WEIGHTS["pix2pix"].keys():

    if "vacv" == virus:
        size = 5888
    else:
        size = 2048

    if "hadv" in virus:
        threshold = -0.9
    else:
        threshold = -0.8

    if "2ch" in virus:
        ch_in = [0, 1]
    else:
        ch_in = [0]

    masks = np.load(
        f"/bigdata/casus/MLID/maria/VIRVS_data/masks/masks_nuc_{virus[:4]}_test.npy"
    )

    dataloader = NpyDataloader(
        path=DATASETS[virus],
        im_size=size,
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
            test_metrics = eval(dataloader, generator, masks, threshold)

            for key, value in test_metrics.items():
                print(f"{key}: {round(value, 3)}")
        else:
            seeds = [42, 43, 44]
            all_seeds_metrics = defaultdict(list)
            generator = Generator(size, ch_in=ch_in, ch_out=1, apply_dropout=dropout)
            generator.load_weights(f"{BASE_PATH}/{WEIGHTS[model][virus]}")
            for seed in seeds:
                tf.keras.utils.set_random_seed(seed)
                test_metrics = eval(dataloader, generator, masks, threshold)
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
