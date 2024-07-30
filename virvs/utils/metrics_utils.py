import numpy as np
from skimage.metrics import structural_similarity


def nmae(y_pred, y_real):
    return np.sum(np.sqrt((y_pred - y_real) ** 2)) / np.sum(np.abs(y_real))


def peak_signal_noise_ratio(y_pred, y_real, data_range):
    err = np.mean(((y_pred - y_real) ** 2))
    return 10 * np.log10((data_range**2) / err)


def calculate_metrics(
    y_pred_batch,
    y_real_batch,
):
    y_pred_batch = np.array(y_pred_batch)
    y_real_batch = np.array(y_real_batch)

    metrics = {
        "mse": np.mean(((y_pred_batch - y_real_batch) ** 2)),
        "nmae": np.mean(
            [
                np.sum(np.abs(y_pred - y_real)) / np.sum(np.abs(y_real))
                for y_pred, y_real in zip(y_pred_batch, y_real_batch)
            ]
        ),
        "psnr": np.mean(
            [
                peak_signal_noise_ratio(y_pred, y_real, data_range=2)
                for y_pred, y_real in zip(y_pred_batch, y_real_batch)
            ]
        ),
        "ssim": np.mean(
            [
                structural_similarity(
                    np.squeeze(y_pred),
                    np.squeeze(y_real),
                    data_range=2,
                )
                for y_pred, y_real in zip(y_pred_batch, y_real_batch)
            ]
        ),
    }

    return metrics
