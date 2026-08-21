"""Cycle GAN implementation link: https://keras.io/examples/generative/cyclegan/"""

from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import MeanSquaredError, Reduction
from tensorflow.keras.optimizers import Adam
from .pix2pix import get_unet_generator, get_patchgan_discriminator

#CycleGAN configuration
KERNEL_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
GAMMA_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor): #pylint: disable=arguments-differ
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        #pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config

def residual_block(
        out,
        activation,
        kernel_initializer=KERNEL_INIT,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        normalization=tfa.layers.InstanceNormalization(gamma_initializer=GAMMA_INIT),
        use_bias=False,
    ):
    """Residual block (with skip connection) in ResNet"""
    dim = out.shape[-1]
    input_tensor = out

    out = ReflectionPadding2D()(input_tensor)
    out = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(out)
    out = normalization(out)
    out = activation(out)
    out = ReflectionPadding2D()(out)
    out = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(out)
    out = normalization(out)
    out = layers.add([input_tensor, out])
    return out


def downsample(
        out,
        filters,
        activation,
        kernel_initializer=KERNEL_INIT,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        normalization=tfa.layers.InstanceNormalization(gamma_initializer=GAMMA_INIT),
        use_bias=False,
    ):
    """Downsample the input layer using stride in convolution"""
    out = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(out)
    out = normalization(out)
    if activation:
        out = activation(out)
    return out


def upsample(
        out,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=KERNEL_INIT,
        normalization=tfa.layers.InstanceNormalization(gamma_initializer=GAMMA_INIT),
        use_bias=False,
    ):
    """Upsample input data using stride in Conv2DTranspose"""
    out = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(out)
    out = normalization(out)
    if activation:
        out = activation(out)
    return out

def get_resnet_generator(
        filters=64,
        num_downsampling_blocks=2,
        num_residual_blocks=9,
        num_upsample_blocks=2,
        gamma_initializer=GAMMA_INIT,
        input_img_dim=(128, 128, 3),
        name=None,
    ):
    """Return GAN generator model"""
    img_input = layers.Input(shape=input_img_dim, name=name + "_img_input")
    out = ReflectionPadding2D(padding=(3, 3))(img_input)
    out = layers.Conv2D(
        filters, (7, 7), kernel_initializer=KERNEL_INIT, use_bias=False)(out)
    out = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(out)
    out = layers.Activation("relu")(out)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        out = downsample(
            out,
            filters=filters,
            activation=layers.Activation("relu"),
            normalization=downsample_norm_fn()
        )

    # Residual blocks
    for i in range(num_residual_blocks):
        if i < 4:
            out = residual_block(
                out, activation=layers.Activation("relu"), normalization=first_half_res_norm_fn())
        else:
            out = residual_block(
                out, activation=layers.Activation("relu"), normalization=second_half_res_norm_fn())
    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        out = upsample(
            out, filters, activation=layers.Activation("relu"), normalization=upsample_norm_fn())

    # Final block
    out = ReflectionPadding2D(padding=(3, 3))(out)
    out = layers.Conv2D(3, (7, 7), padding="valid")(out)
    out = layers.Activation("tanh")(out)

    model = keras.models.Model(img_input, out, name=name)
    return model

def get_discriminator(
        filters=64,
        kernel_initializer=KERNEL_INIT,
        num_downsampling=3,
        input_img_dim=(128, 128, 3),
        name=None
    ):
    """Return GAN discriminator model"""
    img_input = layers.Input(shape=input_img_dim, name=name + "_img_input")
    out = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    out = layers.LeakyReLU(0.2)(out)

    num_filters = filters
    for num_downsample_block in range(num_downsampling):
        num_filters *= 2
        if num_downsample_block < 2:
            out = downsample(
                out,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
                normalization=discriminator_norm_fn()
            )
        else:
            out = downsample(
                out,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
                normalization=discriminator_norm_fn()
            )

    out = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(out)

    model = keras.models.Model(inputs=img_input, outputs=out, name=name)
    return model


@tf.function
def get_resnet_disc_input(x):
    return x


@tf.function
def get_unet_disc_input(x):
    return [x, x]


#pylint: disable=abstract-method
class CycleGan(keras.Model):
    """Cycle GAN implementation using Tensorflow"""

    def __init__(
        self,
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
        disc_input_fn=get_unet_disc_input,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_g = generator_g
        self.gen_f = generator_f
        self.disc_x = discriminator_x
        self.disc_y = discriminator_y
        self.disc_input_fn=disc_input_fn
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.gen_g_optimizer = None
        self.gen_f_optimizer = None
        self.disc_x_optimizer = None
        self.disc_y_optimizer = None
        self.generator_loss_fn = None
        self.discriminator_loss_fn = None
        self.cycle_loss_fn = None
        self.identity_loss_fn = None

    #pylint: disable=arguments-differ
    def compile(
        self,
        gen_g_optimizer,
        gen_f_optimizer,
        disc_x_optimizer,
        disc_y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        #self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        #self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def call(self, inputs):
        _ = inputs

    #pylint: disable=arguments-differ
    def train_step(self, batch_data):
        # x is unstained and y is stained
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # unstained to fake stained
            fake_y = self.gen_g(real_x, training=True)
            # stained to fake unstained -> y2x
            fake_x = self.gen_f(real_y, training=True)

            # Cycle (unstained to fake stained to fake unstained): x -> y -> x
            cycled_x = self.gen_f(fake_y, training=True)
            # Cycle (stained to fake unstained to fake stained) y -> x -> y
            cycled_y = self.gen_g(fake_x, training=True)

             # Identity mapping
            same_x = self.gen_f(real_x, training=True)
            same_y = self.gen_g(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_x(self.disc_input_fn(real_x), training=True)
            disc_fake_x = self.disc_x(self.disc_input_fn(fake_x), training=True)

            disc_real_y = self.disc_y(self.disc_input_fn(real_y), training=True)
            disc_fake_y = self.disc_y(self.disc_input_fn(fake_y), training=True)

            # Generator adverserial loss
            gen_g_loss = self.generator_loss_fn(disc_fake_y)
            gen_f_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_g = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_f = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_g = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_f = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_g = gen_g_loss + cycle_loss_g + id_loss_g
            total_loss_f = gen_f_loss + cycle_loss_f + id_loss_f

            # Discriminator loss
            disc_x_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_g = tape.gradient(total_loss_g, self.gen_g.trainable_variables)
        grads_f = tape.gradient(total_loss_f, self.gen_f.trainable_variables)

        # Get the gradients for the discriminators
        disc_x_grads = tape.gradient(disc_x_loss, self.disc_x.trainable_variables)
        disc_y_grads = tape.gradient(disc_y_loss, self.disc_y.trainable_variables)

        # Update the weights of the generators
        self.gen_g_optimizer.apply_gradients(
            zip(grads_g, self.gen_g.trainable_variables)
        )
        self.gen_f_optimizer.apply_gradients(
            zip(grads_f, self.gen_f.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_x_optimizer.apply_gradients(
            zip(disc_x_grads, self.disc_x.trainable_variables)
        )
        self.disc_y_optimizer.apply_gradients(
            zip(disc_y_grads, self.disc_y.trainable_variables)
        )

        return {
            "G_loss": total_loss_g,
            "F_loss": total_loss_f,
            "D_X_loss": disc_x_loss,
            "D_Y_loss": disc_y_loss,
        }


def downsample_norm_fn():
    """Retrun normalization function for downsampling path"""
    return tfa.layers.InstanceNormalization(gamma_initializer=GAMMA_INIT)

def first_half_res_norm_fn():
    """Return normalization function for first half of resnet residual block"""
    return tfa.layers.InstanceNormalization(gamma_initializer=GAMMA_INIT)

def second_half_res_norm_fn():
    """Return normalization function for second half of resnet residual block"""
    #return CustomInstanceNormalization(kwargs={'second_component':'layer'})
    return tfa.layers.InstanceNormalization(gamma_initializer=GAMMA_INIT)

def upsample_norm_fn():
    """Retrun normalization function for upsampling path"""
    #return CustomInstanceNormalization(kwargs={'second_component':'layer'})
    return tfa.layers.InstanceNormalization(gamma_initializer=GAMMA_INIT)

def discriminator_norm_fn():
    """Return noralization function for discriminator"""
    #return tfa.layers.SpectralNormalization()
    return tfa.layers.InstanceNormalization(gamma_initializer=GAMMA_INIT)


def generic_mse_loss_fn(labels, predictions):
    """Generic MSE"""
    mse = MeanSquaredError()
    loss = mse(labels, predictions)
    return loss

# Define the loss function for the generators
def generator_loss_fn(fake):
    """Loss function for Generator"""
    mse = MeanSquaredError()
    fake_loss = mse(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    """Loss function for Discriminator"""
    mse = MeanSquaredError()
    real_loss = mse(tf.ones_like(real), real)
    fake_loss = mse(tf.zeros_like(fake), fake)
    combined_loss = (real_loss + fake_loss) * 0.5
    return combined_loss

def get_gpu_specific_loss_fns(
        multi_gpu: bool,
        global_batch_size: int
    ) -> Tuple[Callable, Callable, Callable]:
    """Return gen & disc loss functions tailored to either single or multiple gpu"""
    if multi_gpu:
        def multi_gpu_discriminator_loss_fn(real, fake):
            """Combined real and fake loss with multi gpu support"""
            mse = MeanSquaredError(reduction=Reduction.NONE)
            real_loss = mse(tf.ones_like(real), real)
            fake_loss = mse(tf.zeros_like(fake), fake)
            combined_loss = (real_loss + fake_loss) * 0.5
            return tf.nn.compute_average_loss(
                combined_loss, global_batch_size=global_batch_size)

        def multi_gpu_generator_loss_fn(fake):
            """Generator loss function with multi gpu support"""
            mse = MeanSquaredError(reduction=Reduction.NONE)
            fake_loss = mse(tf.ones_like(fake), fake)
            return tf.nn.compute_average_loss(
                fake_loss, global_batch_size=global_batch_size)

        def multi_gpu_adv_loss_fn(labels, predictions):
            """Generic MSE with multi gpu support"""
            mse = MeanSquaredError(reduction=Reduction.NONE)
            loss = mse(labels, predictions)
            return tf.nn.compute_average_loss(
                loss, global_batch_size=global_batch_size)

        gen_loss_fn = multi_gpu_generator_loss_fn
        disc_loss_fn = multi_gpu_discriminator_loss_fn
        mse_gen_loss_fn = multi_gpu_adv_loss_fn
    else:
        gen_loss_fn = generator_loss_fn
        disc_loss_fn = discriminator_loss_fn
        mse_gen_loss_fn = generic_mse_loss_fn

    return gen_loss_fn, disc_loss_fn, mse_gen_loss_fn


def build_cycle_gan(input_img_dim: Tuple[int, int, int], **kwargs) -> Model:
    """Build Cycle GAN using all the components required"""
    use_pix2pix_components = kwargs.get('use_pix2pix_components', False)
    multi_gpu = kwargs.get('multi_gpu', False)
    global_batch_size = kwargs['batch_size']
    normalization = kwargs.get('normalization', 'instancenorm')
    gen_loss_fn, disc_loss_fn, mse_gen_loss_fn = get_gpu_specific_loss_fns(
        multi_gpu=multi_gpu, global_batch_size=global_batch_size)

    if not use_pix2pix_components:
        # Get the generators
        disc_input_fn = get_resnet_disc_input
        gen_g = get_resnet_generator(name="generator_G", input_img_dim=input_img_dim)
        gen_f = get_resnet_generator(name="generator_F", input_img_dim=input_img_dim)
        # Get the discriminators
        disc_x = get_discriminator(name="discriminator_X", input_img_dim=input_img_dim)
        disc_y = get_discriminator(name="discriminator_Y", input_img_dim=input_img_dim)
    else:
        disc_input_fn = get_unet_disc_input
         # Get the generators
        gen_g = get_unet_generator(
            output_channels=3, norm_type=normalization, name="generator_G")
        gen_f = get_unet_generator(
            output_channels=3, norm_type=normalization, name="generator_F")
        #print(input_img_dim)
        # Get the discriminators
        disc_x = get_patchgan_discriminator(
            norm_type=normalization, target=True, name="discriminator_X")
        disc_y = get_patchgan_discriminator(
            norm_type=normalization, target=True, name="discriminator_Y")

    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_g=gen_g,
        generator_f=gen_f,
        discriminator_x=disc_x,
        discriminator_y=disc_y,
        disc_input_fn=disc_input_fn
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_g_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        gen_f_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        disc_x_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        disc_y_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=gen_loss_fn,
        disc_loss_fn=disc_loss_fn,
        cycle_loss_fn=mse_gen_loss_fn,
        identity_loss_fn=mse_gen_loss_fn
    )
    ##print(cycle_gan_model.gen_g.summary())
    ##print(cycle_gan_model.disc_x.summary())
    return cycle_gan_model


def main():
    """Main function to test model size"""
    cyclegan = build_cycle_gan(
        input_img_dim=(256, 256, 3),
        batch_size=64,
        use_pix2pix_components=True,
        normalization='instancenorm'
    )
    ##print(cyclegan.gen_g.summary())
    ##print(cyclegan.disc_x.summary())

if __name__ == '__main__':
    main()
