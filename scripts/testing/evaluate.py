from collections import defaultdict

import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity
from tqdm import tqdm

from virvs.architectures.pix2pix import Generator
from virvs.data.npy_dataloader import NpyDataloader
from virvs.utils.evaluation_utils import round_to_1
from virvs.utils.metrics_utils import calculate_metrics


def ssim_psnr(pred, label):
    ssim = structural_similarity(
        np.squeeze(pred),
        np.squeeze(label),
        data_range=2,
    )
    mse = np.mean((pred - label) ** 2)
    psnr = 10 * np.log10((2**2) / mse)
    return ssim, psnr


BASE_PATH = "/home/wyrzyk93/VIRVS/outputs/weights/"
DATASETS = {
    "hadv_2ch": "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
    "hadv_1ch": "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
    "vacv": "/bigdata/casus/MLID/maria/VIRVS_data/VACV/processed/test",
    "iav": "/bigdata/casus/MLID/maria/VIRVS_data/IAV/processed/test",
    "hsv": "/bigdata/casus/MLID/maria/VIRVS_data/HSV/processed/test",
    "rv": "/bigdata/casus/MLID/maria/VIRVS_data/RV/processed/test",
}

WEIGHTS = {
    "pix2pix": {
        "hadv_2ch": "model_200000_38527895-6967-4f7b-a5df-eb6564b47ad9.h5",
        "hadv_1ch": "model_200000_63065c3c-b6bd-4a38-a5b9-9d9ae5083129.h5",
        "vacv": "model_200000_8b365816-a14c-4279-885f-f61db645e786.h5",
        "iav": "model_200000_fbc9c491-14a6-4ed4-a02d-daefd99fd60e.h5",
        "hsv": "model_200000_afb8f712-ae29-404e-bcd9-a06786e5952b.h5",
        "rv": "model_200000_964056e1-a1e3-4c4f-9e9e-21112f54453a.h5",
    },
    "unet": {
        "hadv_2ch": "model_200000_633f9e8a-0c3b-4678-9a67-ec351fdf09de.h5",
        "hadv_1ch": "model_200000_b301ae59-b3a9-4faf-9ae1-e3b805712147.h5",
        "vacv": "model_200000_d6b695f9-fa90-4ce2-ba99-958078cabd52.h5",
        "iav": "model_200000_868408c5-7b76-412b-9a8d-faf4812ed96f.h5",
        "hsv": "model_200000_1d7dded0-481b-4a29-b3d5-16899f219b6e.h5",
        "rv": "model_200000_d88881a7-e0c3-455f-bb5d-521cfbb98329.h5",
    },
}


def eval(size, ch_in, dropout, model, virus, threshold):
    generator = Generator(size, ch_in=ch_in, ch_out=1, apply_dropout=dropout)
    generator.load_weights(f"{BASE_PATH}/{WEIGHTS[model][virus]}")

    test_metrics = defaultdict(list)
    for sample in dataloader:
        x, y = sample
        output = np.squeeze(generator(np.expand_dims(x, 0), training=True), 0)
        metrics = calculate_metrics(output, y)
        mask = y > threshold
        fg_ssim, fg_psnr = ssim_psnr(output[mask != 0], y[mask != 0])
        bg_ssim, bg_psnr = ssim_psnr(output[mask == 0], y[mask == 0])
        fg_mse = np.mean((output[mask != 0] - y[mask != 0]) ** 2)
        bg_mse = np.mean((output[mask == 0] - y[mask == 0]) ** 2)

        for k, v in metrics.items():
            test_metrics[k].append(v)
            test_metrics["fg_ssim"].append(fg_ssim)
            test_metrics["fg_psnr"].append(fg_psnr)
            test_metrics["fg_mse"].append(fg_mse)
            test_metrics["bg_ssim"].append(bg_ssim)
            test_metrics["bg_psnr"].append(bg_psnr)
            test_metrics["bg_mse"].append(bg_mse)

    for k, v in test_metrics.items():
        test_metrics[k] = np.mean(np.array(v))

    return test_metrics


for virus in DATASETS.keys():

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

    dataloader = NpyDataloader(
        path=DATASETS[virus],
        im_size=size,
        random_jitter=False,
        crop_type="center",
        ch_in=ch_in,
    )
    for model in ["unet", "pix2pix"]:
        if model == "unet":
            dropout = False
        else:
            dropout = True
        print(model, ", ", virus)
        if model == "unet":

            tf.keras.utils.set_random_seed(42)
            test_metrics = eval(size, ch_in, dropout, model, virus, threshold)

            for key, value in test_metrics.items():
                print(f"{key}: {round(value, 3)}")

        else:
            seeds = [42, 43, 44]

            all_seeds_metrics = defaultdict(list)
            for seed in seeds:
                tf.keras.utils.set_random_seed(seed)
                test_metrics = eval(size, ch_in, dropout, model, virus, threshold)
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
