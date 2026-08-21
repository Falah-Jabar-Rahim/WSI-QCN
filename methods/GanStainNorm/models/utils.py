"""Utility function of models"""

import io
from os.path import join
from typing import List

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard


class CycleGanMonitor(Callback):
    """A callback for CycleGAN to generate and save images after each epoch"""
    #pylint: disable=super-init-not-called
    def __init__(
            self,
            summary_writer,
            val_dataset:tf.data.Dataset,
            output_path:str = './output',
            num_img:int = 4
        ):
        self.summary_writer = summary_writer
        self.num_img = num_img
        self.val_dataset = val_dataset
        self.output_path = output_path

    def on_epoch_end(self, epoch, logs=None):
        img_count = 4
        fig, axs = plt.subplots(self.num_img, img_count, figsize=(12, 12))
        for batch_data in self.val_dataset.take(1):
            unstained_batch, stained_batch = batch_data
        vs_stained_batch = self.model.gen_g(unstained_batch)
        vs_unstained_batch = self.model.gen_f(stained_batch)
        for i in range(self.num_img):
            def render_images(img, prediction, i, ind, title):
                pred = (prediction[i] * 127.5 + 127.5).numpy().astype(np.uint8)
                org_img = (img[i] * 127.5 + 127.5).numpy().astype(np.uint8)
                axs[i, ind].imshow(org_img)
                axs[i, ind+1].imshow(pred)
                axs[i, ind].set_title(f'{title} image')
                axs[i, ind+1].set_title('Translated image')
            render_images(unstained_batch, vs_stained_batch, i, 0, 'unstained')
            render_images(stained_batch, vs_unstained_batch, i, 2, 'stained')
            _ = [axs[i, j].axis("off") for j in range(img_count)]

        img = get_pyplot_img(fig)
        with self.summary_writer.as_default():
            tf.summary.image("Validation output", img, step=epoch)

        plt.savefig(join(self.output_path, f'generated_img_{epoch+1}.png'))
        plt.close()

class Pix2PixMonitor(Callback):
    """A callback for Pix2Pix to generate and save images after each epoch"""
    #pylint: disable=super-init-not-called
    def __init__(
            self,
            summary_writer,
            val_dataset: tf.data.Dataset,
            output_path: str = './output',
            num_img: int = 4
        ):
        self.summary_writer = summary_writer
        self.num_img = num_img
        self.val_dataset = val_dataset
        self.output_path = output_path

    def on_epoch_end(self, epoch, logs=None):
        img_count = 3
        fig, axs = plt.subplots(self.num_img, img_count, figsize=(8, 12)) #(w, h)
        for batch_data in self.val_dataset.take(1):
            unstained_batch, stained_batch = batch_data
        predicted_batch = self.model.generator(unstained_batch)
        for i in range(self.num_img):
            predicted_img = (predicted_batch[i] * 127.5 + 127.5).numpy().astype(np.uint8)
            unstained_img = (unstained_batch[i] * 127.5 + 127.5).numpy().astype(np.uint8)
            stained_img = (stained_batch[i] * 127.5 + 127.5).numpy().astype(np.uint8)
            axs[i, 0].imshow(np.reshape(unstained_img, (512, 512)), cmap='gray')
            axs[i, 1].imshow(predicted_img)
            axs[i, 2].imshow(stained_img)
            axs[i, 0].set_title('Grayscale Input')
            axs[i, 1].set_title('Virtually Normalized')
            axs[i, 2].set_title('Original Stian')
            _ = [axs[i, j].axis("off") for j in range(img_count)]

        img = get_pyplot_img(fig)
        with self.summary_writer.as_default():
            tf.summary.image("Validation output", img, step=epoch)
        plt.savefig(join(self.output_path, f'generated_img_{epoch+1}.png'))
        plt.close()

class ValidationSetEvaluation(Callback):
    """Compute MAE for validation set"""

    def __init__(self, summary_writer, validation_data, model_type):
        super(ValidationSetEvaluation, self).__init__()
        self.summary_writer = summary_writer
        self.validation_data = validation_data
        self.model_type = model_type

    #pylint: disable=signature-differs
    def on_epoch_end(self, epoch, logs):
        _ = epoch
        mae_list = []
        vs_model = self.model.gen_g if self.model_type == 'CycleGan' \
            else self.model.generator
        for unstained_batch, stained_batch in self.validation_data:
            vs_batch = vs_model(unstained_batch)
            mae_list.append(
                tf.reduce_mean(
                    tf.abs(stained_batch - vs_batch)
                ).numpy().astype(np.float32)
            )
        with self.summary_writer.as_default():
            tf.summary.scalar('epoch_val_loss', np.mean(mae_list), step=epoch)
        logs['val_loss'] = np.mean(mae_list)
        print(f' - val_loss: {np.mean(mae_list):.5f}')


def build_callbacks(
        val_dataset: tf.data.Dataset,
        model_type: str='CycleGan',
        val_plot_count: int = 4,
        output_path:str ='./output',
        checkpoint_path: str='./checkpoints',
        logs_path: str='./logs'
    ) -> List[Callback]:
    """Return list of callbacks containing checkpoint and plotting functions"""
    
    img_summary_writer = tf.summary.create_file_writer(join(logs_path, 'train'))
    checkpoint_filepath = join(checkpoint_path, 'epoch.{epoch:03d}')
    # callbacks_list = [ModelCheckpoint(filepath=checkpoint_filepath),
    #                   TensorBoard(log_dir=logs_path)]

    callbacks_list = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_freq="epoch",
        ),
        TensorBoard(log_dir=logs_path),
    ]

    if model_type == 'CycleGan':
        plotter = CycleGanMonitor(
            summary_writer=img_summary_writer,
            val_dataset=val_dataset,
            output_path=output_path,
            num_img=val_plot_count)
    if model_type in ['Pix2Pix', 'UthGan', 'DensePix2Pix']:
        plotter = Pix2PixMonitor(
            summary_writer=img_summary_writer,
            val_dataset=val_dataset,
            output_path=output_path,
            num_img=val_plot_count)
        val_summary_writer = tf.summary.create_file_writer(join(logs_path, 'train'))
        val_eval_callback = ValidationSetEvaluation(
            summary_writer=val_summary_writer,
            validation_data=val_dataset,
            model_type=model_type
        )
        callbacks_list.append(val_eval_callback)
    callbacks_list.append(plotter)

    return callbacks_list

def get_pyplot_img(fig):
    """Returns image ensemble from the plot"""
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    return data.reshape((1, int(height), int(width), -1))

def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for shp in out_shape:
            if shp is None:
                continue
            single_layer_mem *= shp
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory
