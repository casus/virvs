from pathlib import Path
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
    images_dir = Path(output_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the full save path
    save_path = images_dir / f"{prefix}_output_{epoch}_{run_id}.png"
    
    # Save the image
    plt.imsave(str(save_path), output_montage)

    # Log to Neptune if run object is provided
    if run is not None:
        run[f"{prefix}_images"].append(
            File(str(save_path)),
            description=f"Epoch {epoch}, {prefix}",
            name=f"{prefix}_output_{epoch}_{run_id}.png"
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

    weights_dir = Path(output_path) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the full save path
    save_path = weights_dir / f"model_{step}_{run_id}.h5"
    
    # Save model weights
    model.save_weights(str(save_path), save_format='h5')

    # Log to Neptune if run object is provided
    if run is not None:
        run[f"model_weights/{save_path.name}"].upload(str(save_path))