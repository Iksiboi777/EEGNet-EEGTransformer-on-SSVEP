# ## Supplementary Material
# Deep Learning in EEG-Based BCIs: A Comprehensive Review of Transformer Models, Advantages, Challenges, and Applications
#
# ### EEGTransformer Class
#
# This notebook contains a TensorFlow/Keras re-implementation of the `EEGTransformer` class, originally presented in PyTorch. It is designed to be a faithful translation, allowing for integration into a TensorFlow-based pipeline.

# Cell 2: Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ### TensorFlow/Keras EEGTransformer Implementation
#
# This cell contains the complete and corrected class definitions.
# Debugging print statements have been re-enabled. To use them, you must
# disable the XLA compiler in your training script by calling:
# model.compile(..., jit_compile=False)

# Cell 4: PositionalEncoding and EEGTransformer Class Definitions
class PositionalEncoding(layers.Layer):
    """
    Custom Keras layer to create and apply positional encodings.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        This method is called for every forward pass.
        """
        _, num_timepoints, num_channels = inputs.shape
        
        positions = tf.range(start=0, limit=num_timepoints, delta=1, dtype=tf.float32)
        channels = tf.range(start=0, limit=num_channels, delta=1, dtype=tf.float32)
        
        angle_rates = 1 / (10000 ** ((2 * (channels // 2)) / tf.cast(num_channels, tf.float32)))
        angle_rads = positions[:, tf.newaxis] * angle_rates[tf.newaxis, :]
        
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return inputs + tf.cast(pos_encoding, dtype=inputs.dtype)

class EEGTransformer(keras.Model):
    """
    TensorFlow/Keras model for EEG data using a Transformer architecture.
    """
    def __init__(self, output_dim,
                 num_heads, key_dim, ffn_intermediate_dim, dropout_rate=0.1,
                 name='EEGTransformer'):
        super(EEGTransformer, self).__init__(name=name)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ffn_intermediate_dim = ffn_intermediate_dim
        self.dropout_rate = dropout_rate

        self.pos_encoding_layer = PositionalEncoding()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="layer_norm_1")
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="layer_norm_2")
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.flatten = layers.Flatten()
        self.classifier = layers.Dense(self.output_dim, name="classifier")
        
        self.multihead_attn = None
        self.ffn = None


    def build(self, input_shape):
        """
        This method is called once when the model first sees an input.
        """
        # tf.print("--- EEGTransformer.build() called ---")
        # tf.print("Input Shape received by build():", input_shape)
        
        _, num_channels, _ = input_shape

        # tf.print("Inferred num_channels for FFN:", num_channels)

        self.multihead_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name='multihead_attention'
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ffn_intermediate_dim, activation="relu"),
                layers.Dense(num_channels),
            ],
            name="feed_forward_network"
        )
        super().build(input_shape)


    def call(self, inputs, training=False):
        """
        Defines the forward pass of the EEGTransformer model.
        """
        # tf.print("\n--- EEGTransformer.call() start ---")
        # tf.print("1. Initial input shape:", tf.shape(inputs))

        mean = tf.reduce_mean(inputs, axis=2, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=2, keepdims=True)
        x = (inputs - mean) / (std + 1e-6)

        x = tf.transpose(x, perm=[0, 2, 1])
        # tf.print("2. Shape after transpose (should be B, T, C):", tf.shape(x))
        
        x = self.pos_encoding_layer(x)
        # tf.print("3. Shape after positional encoding:", tf.shape(x))

        attn_output = self.multihead_attn(query=x, value=x, key=x)
        if training:
            attn_output = self.dropout1(attn_output, training=training)
        
        x = self.norm1(x + attn_output)
        # tf.print("4. Shape after attention + norm (x):", tf.shape(x))

        ffn_output = self.ffn(x)
        if training:
            ffn_output = self.dropout2(ffn_output, training=training)
        
        # tf.print("5. Shape of ffn_output:", tf.shape(ffn_output))
        # tf.print("   Shape of x before final add:", tf.shape(x))
        
        x = self.norm2(x + ffn_output)
        # tf.print("6. Shape after FFN + norm:", tf.shape(x))

        x = self.flatten(x)
        output = self.classifier(x)
        # tf.print("--- EEGTransformer.call() end ---\n")
        return output
