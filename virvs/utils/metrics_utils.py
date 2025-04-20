import numpy as np
from skimage.metrics import structural_similarity


def nmae(y_pred, y_real):
    """Calculates the Normalized Mean Absolute Error (NMAE) between predicted and real values.

    Args:
        y_pred (numpy.ndarray): Predicted values.
        y_real (numpy.ndarray): Ground truth values.

    Returns:
        float: NMAE value computed as sum of absolute errors divided by sum of absolute real values.
    """
    return np.sum(np.sqrt((y_pred - y_real) ** 2)) / np.sum(np.abs(y_real))


def peak_signal_noise_ratio(y_pred, y_real, data_range):
    """Calculates the Peak Signal-to-Noise Ratio (PSNR) between images.

    Args:
        y_pred (numpy.ndarray): Predicted image values.
        y_real (numpy.ndarray): Ground truth image values.
        data_range (float): The possible range of the image values (e.g., 255 for 8-bit images).

    Returns:
        float: PSNR value in decibels.
    """
    err = np.mean(((y_pred - y_real) ** 2))
    return 10 * np.log10((data_range**2) / err)


def calculate_metrics(y_pred_batch, y_real_batch):
    """Calculates various image quality metrics for a batch of predictions.

    Args:
        y_pred_batch (array-like): Batch of predicted images.
        y_real_batch (array-like): Batch of ground truth images.

    Returns:
        dict: Dictionary containing the following metrics:
            - mse: Mean Squared Error
            - nmae: Normalized Mean Absolute Error
            - psnr: Peak Signal-to-Noise Ratio
            - ssim: Structural Similarity Index Measure

    Note:
        Assumes image data range of 2 (for PSNR and SSIM calculations).
        Inputs are converted to numpy arrays if they aren't already.
    """
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