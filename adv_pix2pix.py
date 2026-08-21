"""Implement advanced Pix2Pix"""


from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, initializers
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Dropout, LeakyReLU)
#from tensorflow.keras.layers import add as tf_add
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

from .pix2pix import (Pix2Pix, get_gpu_specific_loss_fns,
                      get_patchgan_discriminator)

KERNEL_INIT = initializers.RandomNormal(mean=0.0, stddev=0.02)
GAMMA_INIT = initializers.RandomNormal(mean=0.0, stddev=0.02)

#pylint: disable=abstract-method
class DensePix2Pix(Pix2Pix):
    """DensePix2Pix inpired by dense UNet

    Publication link:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8718619&tag=1
    """
    def __init__(self, *args, **kwargs):
        super(DensePix2Pix, self).__init__(**kwargs)
        _ = args

def dense_downsample(
        var_x,
        filters: int,
        size: int,
        norm_type: str='batchnorm',
        apply_norm: bool=True):
    """Downsampling block with normalization and activation"""
    out = Conv2D(filters, size, strides=1, padding='same',
                    kernel_initializer=KERNEL_INIT, use_bias=False)(var_x)
    if apply_norm:
        out = BatchNormalization()(out) if norm_type=='batchnorm' else\
            InstanceNormalization(gamma_initializer=GAMMA_INIT)(out)
    out = LeakyReLU()(out)
    first_conv_out = Concatenate()([out, var_x])

    out = Conv2D(filters, size, strides=1, padding='same',
                    kernel_initializer=KERNEL_INIT, use_bias=False)(first_conv_out)
    if apply_norm:
        out = BatchNormalization()(out) if norm_type=='batchnorm' else\
            InstanceNormalization(gamma_initializer=GAMMA_INIT)(out)
    out = LeakyReLU()(out)
    second_conv_out = Concatenate()([out, first_conv_out, var_x])

    out = Conv2D(filters, 1, strides=2, padding='same',
                    kernel_initializer=KERNEL_INIT, use_bias=False)(second_conv_out)
    if apply_norm:
        out = BatchNormalization()(out) if norm_type=='batchnorm' else\
            InstanceNormalization(gamma_initializer=GAMMA_INIT)(out)
    out = LeakyReLU()(out)

    return out

def dense_upsample(
        var_x,
        filters: int,
        filters_secondary: int,
        size: int,
        norm_type: str='batchnorm',
        apply_dropout: bool=False):
    """Upsampling block with normalization and activation"""
    out = Conv2DTranspose(filters, size, strides=2, padding='same',
                    kernel_initializer=KERNEL_INIT, use_bias=False)(var_x)
    out = BatchNormalization()(out) if norm_type=='batchnorm' else\
            InstanceNormalization(gamma_initializer=GAMMA_INIT)(out)
    out = LeakyReLU()(out)

    upsample_out = out
    out = Conv2D(filters_secondary, 1, strides=1, padding='same',
                    kernel_initializer=KERNEL_INIT, use_bias=False)(out)
    out = BatchNormalization()(out) if norm_type=='batchnorm' else\
        InstanceNormalization(gamma_initializer=GAMMA_INIT)(out)
    out = LeakyReLU()(out)
    first_conv_out = Concatenate()([out, upsample_out])

    out = Conv2D(filters_secondary, size, strides=1, padding='same',
                    kernel_initializer=KERNEL_INIT, use_bias=False)(first_conv_out)
    out = BatchNormalization()(out) if norm_type=='batchnorm' else\
        InstanceNormalization(gamma_initializer=GAMMA_INIT)(out)
    out = LeakyReLU()(out)
    second_conv_out = Concatenate()([out, first_conv_out])

    out = Conv2D(filters_secondary, size, strides=1, padding='same',
                    kernel_initializer=KERNEL_INIT, use_bias=False)(second_conv_out)
    out = BatchNormalization()(out) if norm_type=='batchnorm' else\
        InstanceNormalization(gamma_initializer=GAMMA_INIT)(out)
    out = LeakyReLU()(out)
    third_conv_out = Concatenate()([out, second_conv_out, upsample_out])

    out = Conv2D(filters_secondary, 1, strides=1, padding='same',
                    kernel_initializer=KERNEL_INIT, use_bias=False)(third_conv_out)
    out = BatchNormalization()(out) if norm_type=='batchnorm' else\
        InstanceNormalization(gamma_initializer=GAMMA_INIT)(out)
    out = LeakyReLU()(out)
    if apply_dropout:
        out = Dropout(0.5)(out)

    return out


def get_dense_unet_generator(
        output_channels: int,
        input_channels: int=3,
        norm_type: str='batchnorm',
        name: str='dense_pix2pix_generator'
    ):
    """Modified U-Net generator model (https://arxiv.org/abs/1611.07004)"""

    inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
    var_x = inputs

    d_1 = dense_downsample(var_x, 64, 4, norm_type, apply_norm=False)  # (bs, 256, 256, 64)
    d_2 = dense_downsample(d_1, 128, 4, norm_type)  # (bs, 128, 128, 128)
    d_3 = dense_downsample(d_2, 256, 4, norm_type)  # (bs, 64, 64, 256)
    d_4 = dense_downsample(d_3, 512, 4, norm_type)  # (bs, 32, 32, 512)
    d_5 = dense_downsample(d_4, 512, 4, norm_type)  # (bs, 16, 16, 512)
    d_6 = dense_downsample(d_5, 512, 4, norm_type)  # (bs, 8, 8, 512)
    d_7 = dense_downsample(d_6, 512, 4, norm_type)  # (bs, 4, 4, 512)
    d_8 = dense_downsample(d_7, 512, 4, norm_type)  # (bs, 1, 1, 512)

    u_1 = dense_upsample(d_8, 512, 512, 4, norm_type, apply_dropout=True)  # (bs, 2, 2, 1024)
    skip_res = Concatenate()([u_1, d_7])
    u_2 = dense_upsample(skip_res, 512, 512, 4, norm_type, apply_dropout=True)  # (bs, 4, 4, 1024)
    skip_res = Concatenate()([u_2, d_6])
    u_3 = dense_upsample(skip_res, 512, 256, 4, norm_type, apply_dropout=True)  # (bs, 8, 8, 1024)
    skip_res = Concatenate()([u_3, d_5])
    u_4 = dense_upsample(skip_res, 512, 256, 4, norm_type)  # (bs, 16, 16, 1024)
    skip_res = Concatenate()([u_4, d_4])
    u_5 = dense_upsample(skip_res, 256, 128, 4, norm_type)  # (bs, 32, 32, 512)
    skip_res = Concatenate()([u_5, d_3])
    u_6 = dense_upsample(skip_res, 128, 64, 4, norm_type)  # (bs, 64, 64, 256)
    skip_res = Concatenate()([u_6, d_2])
    u_7 = dense_upsample(skip_res, 64, 32, 4, norm_type)  # (bs, 128, 128, 128)
    skip_res = Concatenate()([u_7, d_1])

    var_x = tf.keras.layers.Conv2DTranspose(
                output_channels, 4, strides=2,
                padding='same', kernel_initializer=KERNEL_INIT,
                activation='tanh'
            )(skip_res)

    return tf.keras.Model(inputs=inputs, outputs=var_x, name=name)

def build_dense_pix2pix(input_img_dim: Tuple[int, int, int], **kwargs) -> Model:
    """Build Pix2pix model using all the required components"""
    _ = input_img_dim
    multi_gpu = kwargs.get('multi_gpu', False)
    global_batch_size = kwargs['batch_size']
    gen_loss_fn, disc_loss_fn = get_gpu_specific_loss_fns(multi_gpu, global_batch_size, True)
    unet_dense_generator = get_dense_unet_generator(
        output_channels=3, input_channels=1, norm_type=kwargs['normalization'])
    patchgan_discriminator = get_patchgan_discriminator(
        output_channels=3, input_channels=1, norm_type=kwargs['normalization'])
    dense_pix2pix_model = DensePix2Pix(
        generator=unet_dense_generator,
        discriminator=patchgan_discriminator,
        lambda_value=100
    )
    dense_pix2pix_model.compile(
        generator_loss=gen_loss_fn,
        discriminator_loss=disc_loss_fn,
        generator_optimizer=Adam(2e-4, beta_1=0.5),
        discriminator_optimizer=Adam(2e-4, beta_1=0.5),
    )
    print(dense_pix2pix_model.generator.summary())
    print(dense_pix2pix_model.discriminator.summary())
    return dense_pix2pix_model

def main():
    """Main function to test model size"""
    _ = build_dense_pix2pix((512, 512, 3), batch_size=64, normalization='instancenorm')

if __name__ == '__main__':
    main()
