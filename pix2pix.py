"""Pix2Pix implementation inspired by
https://github.com/tensorflow/examples/blob/master/
tensorflow_examples/models/pix2pix/pix2pix.py"""


from typing import Callable, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import Model, initializers, layers
from tensorflow.keras.layers import (Add, Dense, Dropout, Flatten,
                                     LayerNormalization, MultiHeadAttention,
                                     Reshape)
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

from .custom_layers import CustomInstanceNormalization, PatchEmbeddings

KERNEL_INIT = initializers.RandomNormal(mean=0.0, stddev=0.02)
GAMMA_INIT = initializers.RandomNormal(mean=0.0, stddev=0.02)

def downsample(
        filters: int,
        size: int,
        norm_type: str='batchnorm',
        apply_norm: bool=True,
        double_conv: bool=False):
    """Downsampling block with normalization and activation"""
    result = tf.keras.Sequential()
    ## Experimental Convolution Starts
    if double_conv:
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                kernel_initializer=KERNEL_INIT, use_bias=False))
        result.add(InstanceNormalization(gamma_initializer=GAMMA_INIT))
        result.add(tf.keras.layers.LeakyReLU())
    ## Experimental Convolution Ends

    result.add(
         tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=KERNEL_INIT, use_bias=False))
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization(gamma_initializer=GAMMA_INIT))
        elif norm_type.lower() == 'batchinstancenorm':
            result.add(CustomInstanceNormalization(second_component='batch'))
        elif norm_type.lower() == 'layerinstancenorm':
            result.add(CustomInstanceNormalization(second_component='layer'))
        else:
            raise RuntimeError(f'Normalization type {norm_type.lower()} not recognized')
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(
        filters: int,
        size: int,
        norm_type: str='batchnorm',
        apply_dropout: bool=False,
        double_conv: bool=False):
    """Upsampling block with normalization and activation"""
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=KERNEL_INIT,
                                        use_bias=False))
    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization(gamma_initializer=GAMMA_INIT))
    elif norm_type.lower() == 'batchinstancenorm':
        result.add(CustomInstanceNormalization(second_component='batch'))
    elif norm_type.lower() == 'layerinstancenorm':
        result.add(CustomInstanceNormalization(second_component='layer'))
    else:
        raise RuntimeError(f'Normalization type {norm_type.lower()} not recognized')

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())

    ## Experimental Convolution Starts
    if double_conv:
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                kernel_initializer=KERNEL_INIT, use_bias=False))
        result.add(InstanceNormalization(gamma_initializer=GAMMA_INIT))
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
    ## Experimental Convolution Ends
    return result

def mlp(var_x, hidden_units: Sequence[int], dropout_rate: int):
    """Multi-layer perceptron"""
    for units in hidden_units:
        var_x = Dense(units, activation=tf.nn.gelu)(var_x)
        var_x = Dropout(dropout_rate)(var_x)
    return var_x

def get_unet_generator(
        output_channels: int,
        input_channels: int=3,
        norm_type: str='batchnorm',
        name: str='pix2pix_generator'
    ):
    """Modified U-Net generator model (https://arxiv.org/abs/1611.07004)"""

    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
        downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
        downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
        downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
        downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
        downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
        downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
        upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
        upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
        upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
    ]
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=KERNEL_INIT,
        activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
    var_x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        var_x = down(var_x)
        skips.append(var_x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up_level, skip in zip(up_stack, skips):
        var_x = up_level(var_x)
        var_x = concat([var_x, skip])

    var_x = last(var_x)

    return tf.keras.Model(inputs=inputs, outputs=var_x, name=name)

def get_patchgan_discriminator(
        output_channels: int=3,
        input_channels: int=3,
        norm_type: str='batchnorm',
        target: bool=True,
        name: str='pg_discriminator'):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004)"""
    inp = tf.keras.layers.Input(shape=[None, None, input_channels], name='input_image')
    var_x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, output_channels], name='target_image')
        var_x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, norm_type, apply_norm=False, double_conv=False)(var_x)
    # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type, double_conv=False)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type, double_conv=False)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=KERNEL_INIT,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization(gamma_initializer=GAMMA_INIT)(conv)
    elif norm_type.lower() == 'batchinstancenorm':
        norm1 = CustomInstanceNormalization(second_component='batch')(conv)
    elif norm_type.lower() == 'layerinstancenorm':
        norm1 = CustomInstanceNormalization(second_component='layer')(conv)
    else:
        raise RuntimeError(f'Normalization type {norm_type.lower()} not recognized')

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=KERNEL_INIT)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last, name=name)

#pylint: disable=abstract-method
class Pix2Pix(Model):
    """Pix2Pix model implementation"""

    def __init__(
            self,
            generator,
            discriminator,
            lambda_value,
            ):
        super(Pix2Pix, self).__init__()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = Adam(2e-4, beta_1=0.5)
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_value = lambda_value
        self.generator_loss = None
        self.discriminator_loss = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    #pylint: disable=arguments-differ
    def call(self, inputs):
        _ = inputs

    #pylint: disable=arguments-differ
    def compile(
            self,
            generator_loss,
            discriminator_loss,
            generator_optimizer,
            discriminator_optimizer,
            ):
        super(Pix2Pix, self).compile()
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    #pylint: disable=arguments-differ
    def train_step(self, batch_data):
        """One train step over the generator and discriminator model"""
        #input_image, target_image = batch_data[:, 0, ...], batch_data[:, 1, ...]
        input_image, target_image = batch_data

        input_image = tf.ensure_shape(

            input_image,

            [None, 512, 512, 1],

        )

        target_image = tf.ensure_shape(

            target_image,

            [None, 512, 512, 3],

        )

        input_image = tf.cast(input_image, tf.float32)

        target_image = tf.cast(target_image, tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator(
                [input_image, target_image], training=True)
            disc_generated_output = self.discriminator(
                [input_image, gen_output], training=True)

            gen_loss = self.generator_loss(
                disc_generated_output, gen_output, target_image, self.lambda_value)
            disc_loss = self.discriminator_loss(
                disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(
            generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(
            discriminator_gradients, self.discriminator.trainable_variables))

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}


def get_uth_generator(output_channels: int, norm_type: str='batchnorm'):
    """UTH: U-Net Tranformer Hybird generator model"""
    encoder_stack = [
        downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
        downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
    ]

    decoder_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 64, 64, 512)
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 128, 128, 512)
        #upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 128, 128, 1024)
    ]

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    var_x = inputs

    #Downsampling through the model
    skips = []
    for down in encoder_stack:
        var_x = down(var_x)
        skips.append(var_x)

    #Patch Embedding Configurations
    mlp_hidden_unit = [1024, 512] #[4096, 768]
    transformer_layers = 4 #12
    hidden_size = 512 #768
    patch_size = 2**len(encoder_stack)
    img_size = 256

    patch_embeddings = PatchEmbeddings(
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=hidden_size
    )(var_x)

    encodings = patch_embeddings
    for _ in range(transformer_layers):
        #Layer normalization
        var_x1 = LayerNormalization(epsilon=1e-6)(encodings)
        #Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=512, dropout=0.1 #key_dim=768
        )(var_x1, var_x1)
        #Skip connection
        var_x2 = Add()([attention_output, encodings])
        # Layer normalization 2.
        var_x3 = LayerNormalization(epsilon=1e-6)(var_x2)
        # MLP.
        var_x3 = mlp(var_x3, hidden_units=mlp_hidden_unit, dropout_rate=0.1)
        # Skip connection 2.
        encodings = layers.Add()([var_x3, var_x2])

    encoding_reshape = (int(img_size/patch_size), int(img_size/patch_size), hidden_size)
    var_x = Reshape(target_shape=encoding_reshape)(encodings)

    concat = tf.keras.layers.Concatenate()
    skips = reversed(skips[:-1])
    #Upsampling and establishing the skip connections
    for up_level, skip in zip(decoder_stack, skips):
        var_x = up_level(var_x)
        var_x = concat([var_x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=KERNEL_INIT,
        activation='tanh')  # (bs, 256, 256, 3)

    var_x = last(var_x)

    return tf.keras.Model(inputs=inputs, outputs=var_x, name='uth_generator')


def get_uthgan_discriminator(norm_type: str='batchnorm', target: bool=True):
    """A discriminator based on convolution-transformer hybrid"""

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    var_x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        var_x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, norm_type, False)(var_x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

    mlp_hidden_unit = [1024, 512] #[4096, 768]
    transformer_layers = 2 #10 #12
    hidden_size = 512 #768
    patch_size = 2**3
    img_size = 256

    patch_embeddings = PatchEmbeddings(
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=hidden_size
    )(down3)

    encodings = patch_embeddings
    for _ in range(transformer_layers):
        #Layer normalization
        var_x1 = LayerNormalization(epsilon=1e-6)(encodings)
        #Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=512, dropout=0.1 #head=12, key_dim=768
        )(var_x1, var_x1)
        #Skip connection
        var_x2 = Add()([attention_output, encodings])
        # Layer normalization 2.
        var_x3 = LayerNormalization(epsilon=1e-6)(var_x2)
        # MLP.
        var_x3 = mlp(var_x3, hidden_units=mlp_hidden_unit, dropout_rate=0.1)
        # Skip connection 2.
        encodings = layers.Add()([var_x3, var_x2])

    representation = LayerNormalization(epsilon=1e-6)(encodings)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=[128, 64], dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(1)(features)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=logits, name='uth-discriminator')
    else:
        return tf.keras.Model(inputs=inp, outputs=logits, name='uth-discriminator')

class UthGan(Pix2Pix):
    """UnetTransformerGAN a Pix2Pix with modified generator"""
    def __init__(self, *args, **kwargs):
        super(UthGan, self).__init__(**kwargs)
        _ = args

def discriminator_loss_fn(disc_real_output, disc_generated_output):
    """Combined real and generated loss"""
    bc_loss = BinaryCrossentropy(from_logits=True)
    real_loss = bc_loss(
        tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = bc_loss(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss_fn(disc_generated_output, gen_output, target, lambda_value):
    """Generator loss as the sum of weighted GAN loss and L1 loss of generated ouput"""
    bc_loss = BinaryCrossentropy(from_logits=True)
    gan_loss = bc_loss(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (lambda_value * l1_loss)
    return total_gen_loss

def get_gpu_specific_loss_fns(
        multi_gpu: bool,
        global_batch_size: int,
        patch_gan_disc: bool = False
    ) -> Tuple[Callable, Callable]:
    """Return gen & disc loss functions tailored to either single or multiple gpu"""
    if multi_gpu:
        def multi_gpu_discriminator_loss_fn(disc_real_output, disc_generated_output):
            """Combined real and generated loss for multi gpu training"""
            bc_loss = BinaryCrossentropy(from_logits=True, reduction=Reduction.NONE)
            real_loss = bc_loss(
                tf.ones_like(disc_real_output), disc_real_output)

            generated_loss = bc_loss(tf.zeros_like(
                disc_generated_output), disc_generated_output)

            total_disc_loss = real_loss + generated_loss
            return tf.nn.compute_average_loss(
                total_disc_loss, global_batch_size=global_batch_size)

        def multi_gpu_generator_loss_fn(disc_generated_output, gen_output, target, lambda_value):
            """Generator loss as the sum of weighted GAN loss and L1 loss of generated ouput"""
            bc_loss = BinaryCrossentropy(from_logits=True, reduction=Reduction.NONE)
            gan_loss = bc_loss(tf.ones_like(
                disc_generated_output), disc_generated_output)

            # mean absolute error
            l1_loss = tf.reduce_mean(tf.abs(target - gen_output), [1, 2, 3])
            # A check to reduce (via mean) h, w to a single point in case of patchgan discriminator
            # This is to ensure that gan_loss and l1_loss have the same dimensions
            gan_loss = tf.reduce_mean(gan_loss, [1, 2]) if patch_gan_disc else gan_loss
            total_gen_loss = gan_loss + (lambda_value * l1_loss)
            return tf.nn.compute_average_loss(
                total_gen_loss, global_batch_size=global_batch_size)

        gen_loss_fn = multi_gpu_generator_loss_fn
        disc_loss_fn = multi_gpu_discriminator_loss_fn
    else:
        gen_loss_fn = generator_loss_fn
        disc_loss_fn = discriminator_loss_fn
    return gen_loss_fn, disc_loss_fn

def build_pix2pix(input_img_dim: Tuple[int, int, int], **kwargs) -> Model:
    """Build Pix2pix model using all the required components"""
    _ = input_img_dim
    multi_gpu = kwargs.get('multi_gpu', False)
    global_batch_size = kwargs['batch_size']
    gen_loss_fn, disc_loss_fn = get_gpu_specific_loss_fns(multi_gpu, global_batch_size, True)
    unet_generator = get_unet_generator(
        output_channels=3, input_channels=1, norm_type=kwargs['normalization'])
    patchgan_discriminator = get_patchgan_discriminator(
        output_channels=3, input_channels=1, norm_type=kwargs['normalization'])
    pix2pix_model = Pix2Pix(
        generator=unet_generator,
        discriminator=patchgan_discriminator,
        lambda_value=100
    )
    pix2pix_model.compile(
        generator_loss=gen_loss_fn,
        discriminator_loss=disc_loss_fn,
        generator_optimizer=Adam(2e-4, beta_1=0.5),
        discriminator_optimizer=Adam(2e-4, beta_1=0.5),
    )
    #print(pix2pix_model.generator.summary())
    #print(pix2pix_model.discriminator.summary())
    return pix2pix_model

def build_uthgan(input_img_dim: Tuple[int, int, int], **kwargs) -> Model:
    """Build UthGan model using all the required components"""
    _ = input_img_dim
    multi_gpu = kwargs.get('multi_gpu', False)
    global_batch_size = kwargs['batch_size']
    gen_loss_fn, disc_loss_fn = get_gpu_specific_loss_fns(multi_gpu, global_batch_size, False)
    uth_generator = get_uth_generator(output_channels=3, norm_type=kwargs['normalization'])
    uth_discriminator = get_uthgan_discriminator(norm_type=kwargs['normalization'])
    uthgan_model = UthGan(
        generator=uth_generator,
        discriminator=uth_discriminator,
        lambda_value=100
    )
    uthgan_model.compile(
        generator_loss=gen_loss_fn,
        discriminator_loss=disc_loss_fn,
        generator_optimizer=Adam(2e-4, beta_1=0.5),
        discriminator_optimizer=Adam(2e-4, beta_1=0.5),
    )
    print(uthgan_model.generator.summary())
    print(uthgan_model.discriminator.summary())
    return uthgan_model

def main():
    """Main function to test model size"""
    pix2pix = build_pix2pix((256, 256, 3), batch_size=64, normalization='batchnorm')
    print(pix2pix.generator.summary())
    print(pix2pix.discriminator.summary())

if __name__ == '__main__':
    main()
