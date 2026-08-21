"""Implement custom normalization functions"""

from typing import Any, Union

import tensorflow as tf
from tensorflow.keras import layers


class CustomInstanceNormalization(layers.Layer):
    """Layer or Batch Instance Normalization Layer https://arxiv.org/abs/1805.07925"""

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__()
        self.epsilon = epsilon
        if kwargs.get('second_component', 'layer') == 'layer':
            self.second_component_axis = [1, 2, 3]
        else:
            self.second_component_axis = [0, 1, 2]
        self.scale: Union[tf.Variable, Any]
        self.offset: Union[tf.Variable, Any]
        self.rho: Union[tf.Variable, Any]

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

        self.rho = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

    def call(self, inputs, **kwargs):
        _ = kwargs
        #instance normalization
        ins_mean, ins_variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        ins_inv = tf.math.rsqrt(ins_variance + self.epsilon)
        ins_normalized = (inputs - ins_mean) * ins_inv
        #layer normalization axis [1, 2, 3] | batch normalization [0, 1, 2]
        sc_mean, sc_variance = tf.nn.moments(inputs, axes=self.second_component_axis, keepdims=True)
        sc_inv = tf.math.rsqrt(sc_variance + self.epsilon)
        sc_normalized = (inputs - sc_mean) * sc_inv

        normalized = self.rho * ins_normalized + (1 - self.rho) * sc_normalized

        return self.scale * normalized + self.offset

    def get_config(self):
        config = super().get_config().copy()
        return config

class PatchEmbeddings(layers.Layer):
    """Custom layer to divde image into smaller patches"""
    def __init__(self, img_size: int, patch_size: int, hidden_size: int):
        super(PatchEmbeddings, self).__init__()
        self.real_img_patch_size = patch_size
        #self.patch_size = patch_size // 8 # =1
        self.hidden_size = hidden_size
        self.n_patches = int((img_size / patch_size) * (img_size / patch_size)) #=patch_size=16
        self.position_embeddings: Union[tf.Variable, Any]
        self.patch_conv: Union[tf.Variable, Any]

    def build(self, input_shape):
        _ = input_shape
        self.position_embeddings = self.add_weight(
            name='position_embeddings',
            shape=(1, self.n_patches, self.hidden_size),
            initializer='zeros',
            trainable=True)
        self.patch_conv= layers.Conv2D(
            filters=self.hidden_size,
            kernel_size= 1, #self.patch_size,
            strides=1 #(self.patch_size, self.patch_size)
        )

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        patch_embeddings = self.patch_conv(inputs)
        patch_embeddings = tf.reshape(
            patch_embeddings, [batch_size, self.n_patches, self.hidden_size])
        patch_embeddings = patch_embeddings + self.position_embeddings
        patch_embeddings = layers.Dropout(0.1)(patch_embeddings)
        return patch_embeddings
