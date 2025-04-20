from matplotlib import pyplot as plt
from neptune.types import File


def log_metrics(run, metrics_dict, prefix):
    """Logs metrics to a Neptune run.

    Args:
        run (neptune.Run): Neptune run object to log metrics to. If None, no logging occurs.
        metrics_dict (dict): Dictionary of metric names and their values to log.
        prefix (str): Prefix to add to each metric name before logging.
    """
    if run is not None:
        for metric_name, metric_value in metrics_dict.items():
            run[f"{prefix}_" + metric_name].log(metric_value)


def save_output_montage(
    run,
    output_montage,
    epoch,
    output_path,
    run_id,
    prefix,
):
    """Saves and logs an output montage image to disk and Neptune.

    Args:
        run (neptune.Run): Neptune run object to log to. If None, no logging occurs.
        output_montage (numpy.ndarray): Image data to save as a montage.
        epoch (int): Current epoch number for naming purposes.
        output_path (str): Base directory path to save the image.
        run_id (str): Unique identifier for the run for naming purposes.
        prefix (str): Prefix to add to the filename and Neptune log entry.
    """
    plt.imsave(
        f"{output_path}/images/{prefix}_output_{str(epoch)}_{run_id}.png",
        output_montage,
    )

    if run is not None:
        run[f"{prefix}_images"].append(
            File(f"{output_path}/images/{prefix}_output_{str(epoch)}_{run_id}.png"),
            description=f"Epoch {epoch}, {prefix}",
        )


def save_weighs(run, model, step, output_path, run_id):
    """Saves model weights to disk and optionally logs to Neptune.

    Args:
        run (neptune.Run): Neptune run object to log to. If None, no logging occurs.
        model (tf.keras.Model): Model whose weights should be saved.
        step (int): Training step number for naming purposes.
        output_path (str): Base directory path to save the weights.
        run_id (str): Unique identifier for the run for naming purposes.
    """
    model.save_weights(f"{output_path}/weights/model_{str(step)}_{run_id}.h5", True)

    if run is not None:
        run[f"model_weights/model_{str(step)}_{run_id}.h5"].upload(
            f"{output_path}/weights/model_{str(step)}_{run_id}.h5"
        )