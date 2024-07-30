import argparse
import uuid
from collections import defaultdict

import neptune as neptune
import numpy as np
import tensorflow as tf
from skimage.util import montage
from virvs.architectures.pix2pix import Discriminator, Generator
from virvs.configs.utils import (
    create_data_config,
    create_eval_config,
    create_neptune_config,
    create_training_config,
    load_config_from_yaml,
)
from virvs.utils.inference_utils import log_metrics, save_output_montage, save_weighs
from virvs.utils.metrics_utils import calculate_metrics
from virvs.utils.training_utils import prepare_dataset

tf.keras.utils.set_random_seed(42)


def generator_loss(discriminator_generated_output, generator_output, ground_truth):
    gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(discriminator_generated_output), discriminator_generated_output
    )

    l1_loss = tf.reduce_mean(tf.abs(ground_truth - generator_output))
    total_loss = gen_loss + 100 * l1_loss
    return total_loss, gen_loss, l1_loss


def discriminator_loss(
    discriminator_real_output, discriminator_generated_output, disc_weight
):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(discriminator_real_output), discriminator_real_output
    )
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(discriminator_generated_output), discriminator_generated_output
    )
    total_loss = real_loss + generated_loss
    return total_loss * disc_weight


@tf.function
def train_step(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    input_image,
    target,
    disc_weight,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_total_loss = discriminator_loss(
            disc_real_output, disc_generated_output, disc_weight
        )

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_total_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )
    return gen_output, gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", help="Path to the configuration file", required=True
    )
    parser.add_argument("--neptune-token", help="API token for Neptune")

    args = parser.parse_args()

    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    config = load_config_from_yaml(args.config_path)
    data_config = create_data_config(config)
    training_config = create_training_config(config)
    eval_config = create_eval_config(config)
    neptune_config = create_neptune_config(config)

    print("Getting data...")

    assert data_config.train_data_path is not None
    assert data_config.val_data_path is not None

    channels_in = data_config.ch_in
    channels_out = 1

    dataset = prepare_dataset(
        path=data_config.train_data_path,
        im_size=data_config.im_size,
        random_jitter=True,
        ch_in=channels_in,
        ch_out=channels_out,
    )
    val_dataset = prepare_dataset(
        path=data_config.val_data_path,
        im_size=256,
        ch_in=channels_in,
        ch_out=channels_out,
    )

    batch_size = data_config.batch_size

    dataset = dataset.shuffle(5000)
    val_dataset = val_dataset.shuffle(5000)

    dataset = dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    generator = Generator(256, ch_in=channels_in, ch_out=channels_out)
    discriminator = Discriminator(256, ch_in=channels_in, ch_out=channels_out)
    generator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    output_path = eval_config.output_path

    run = None
    if args.neptune_token is not None and neptune_config is not None:
        run = neptune.init_run(
            api_token=args.neptune_token,
            name=neptune_config.name,
            project=neptune_config.project,
        )
        run["config.yaml"].upload(args.config_path)

    print("Starting training...")

    step = 0

    log_freq = eval_config.log_freq
    val_freq = eval_config.val_freq
    max_steps = training_config.max_steps

    run_id = str(uuid.uuid4())
    cumulative_loss = np.zeros(4)
    train_metrics = defaultdict(float)
    while True:

        for batch in dataset:

            batch_x, batch_y = batch

            output, gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss = train_step(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                batch_x,
                batch_y,
                training_config.pix2pix_disc_weight,
            )

            cumulative_loss += np.array(
                [gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss]
            )
            metrics = calculate_metrics(output, batch_y.numpy())
            for k, v in metrics.items():
                train_metrics[k] += v

            if step % log_freq == 0:
                if run is not None:
                    run[f"train_loss_total"].log(
                        (cumulative_loss[0] + cumulative_loss[3]) / log_freq
                    )
                    run[f"train_loss_gen_total"].log((cumulative_loss[0]) / log_freq)
                    run[f"train_loss_gen"].log((cumulative_loss[1]) / log_freq)
                    run[f"train_loss_l1"].log((cumulative_loss[2]) / log_freq)
                    run[f"train_loss_disc_total"].log((cumulative_loss[3]) / log_freq)
                    cumulative_loss = np.zeros(4)
                    log_metrics(run, train_metrics, prefix="train")
                    train_metrics = defaultdict(float)

            if step % val_freq == 0:
                val_metrics = defaultdict(float)

                for n, batch in enumerate(val_dataset):
                    batch_x, batch_y = batch
                    output = generator(batch_x, training=True)

                    metrics = calculate_metrics(output, batch_y.numpy())
                    for k, v in metrics.items():
                        val_metrics[k] += v

                for k, v in val_metrics.items():
                    val_metrics[k] = val_metrics[k] / (n + 1)

                log_metrics(run, val_metrics, prefix="val")
                if len(channels_in) == 2:
                    montage_content = np.concatenate(
                        (
                            output,
                            batch_y,
                            batch_x[..., 0:1],
                            batch_x[..., 1:2],
                        ),
                        axis=2,
                    )
                else:
                    montage_content = np.concatenate(
                        (
                            output,
                            batch_y,
                            batch_x[..., 0:1],
                        ),
                        axis=2,
                    )

                output_montage = montage(np.squeeze(montage_content))
                output_montage = np.uint8(output_montage * 127.5 + 127.5)
                save_output_montage(
                    run=run,
                    output_montage=output_montage,
                    epoch=step,
                    output_path=output_path,
                    run_id=run_id,
                    prefix="val",
                )

            if step == max_steps:
                if run is not None:
                    save_weighs(
                        run=run,
                        model=generator,
                        step=step,
                        output_path=output_path,
                        run_id=run_id,
                    )
                    run.stop()
                exit(0)
            step += 1


if __name__ == "__main__":
    main()
