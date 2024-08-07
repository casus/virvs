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
        "hadv_2ch": "model_100000_316674a2-4299-4a06-b601-5e20f7dd02a6.h5",
        "hadv_1ch": "model_100000_d6792b38-8091-448d-a26a-ef08375b8dbe.h5",
        "vacv": "model_100000_138fd26a-88b9-4d2a-9f96-6a2ffe991364.h5",
        "iav": "model_100000_ef994584-7730-41b9-9a56-75d478bacf02.h5",
        "hsv": "model_100000_52d604f6-25fe-4d42-a212-cef281e3a8b5.h5",
        "rv": "model_100000_a6bd97a2-d889-40a4-a8ec-5c8816797f9d.h5",
    },
    "unet": {
        "hadv_2ch": "model_100000_c0175f01-1e6e-4bec-9f2e-2e5542c16584.h5",
        "hadv_1ch": "model_100000_47b9346c-e143-4bf7-9dce-40aa6f2329e5.h5",
        "vacv": "model_100000_4f91526f-a1b4-4057-926e-073f4ffbef67.h5",
        "iav": "model_100000_26c5b8b9-0c35-4fee-9eac-44a857cebe76.h5",
        "hsv": "model_100000_1b8e6e99-10fa-4221-b53a-b680a65826be.h5",
        "rv": "model_100000_0ca537f4-cbd8-4722-89cb-bdd43107a66b.h5",
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
        test_metrics[k + "_std"] = np.std(np.array(v))

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

        tf.keras.utils.set_random_seed(42)
        test_metrics = eval(size, ch_in, dropout, model, virus, threshold)

        for key, value in test_metrics.items():
            if "std" not in key:
                print(
                    f"{key}: {round(value, 3)}, std: {round(test_metrics[key+'_std'], 3)}"
                )
