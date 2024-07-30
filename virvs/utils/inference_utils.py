from matplotlib import pyplot as plt
from neptune.types import File


def log_metrics(run, metrics_dict, prefix):
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

    model.save_weights(f"{output_path}/weights/model_{str(step)}_{run_id}.h5", True)

    if run is not None:
        run[f"model_weights/model_{str(step)}_{run_id}.h5"].upload(
            f"{output_path}/weights/model_{str(step)}_{run_id}.h5"
        )
