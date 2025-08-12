# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# class PositionalEncoding(layers.Layer):
#     """
#     Custom Keras layer to create and apply positional encodings.
#     """
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def call(self, inputs):
#         """
#         Creates the positional encoding matrix on the fly.
#         """
#         _, num_timepoints, num_features = inputs.shape
        
#         positions = tf.range(start=0, limit=num_timepoints, delta=1, dtype=tf.float32)
#         feature_indices = tf.range(start=0, limit=num_features, delta=1, dtype=tf.float32)
        
#         angle_rates = 1 / (10000 ** ((2 * (feature_indices // 2)) / tf.cast(num_features, tf.float32)))
#         angle_rads = positions[:, tf.newaxis] * angle_rates[tf.newaxis, :]
        
#         sines = tf.sin(angle_rads[:, 0::2])
#         cosines = tf.cos(angle_rads[:, 1::2])
        
#         pos_encoding = tf.concat([sines, cosines], axis=-1)
#         pos_encoding = pos_encoding[tf.newaxis, ...]
        
#         return inputs + tf.cast(pos_encoding, dtype=inputs.dtype)

# class EEGTransformer(keras.Model):
#     """
#     Final TensorFlow/Keras model for EEG data using a Transformer architecture.
#     This version expects input in the shape (Batch, Timepoints, Features).
#     """
#     def __init__(self, output_dim, num_heads, key_dim, ffn_intermediate_dim, dropout_rate=0.1, name='EEGTransformer'):
#         super(EEGTransformer, self).__init__(name=name)
#         self.output_dim = output_dim
#         self.num_heads = num_heads
#         self.key_dim = key_dim
#         self.ffn_intermediate_dim = ffn_intermediate_dim
#         self.dropout_rate = dropout_rate

#         self.pos_encoding_layer = PositionalEncoding()
#         self.norm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(self.dropout_rate)
#         self.norm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout2 = layers.Dropout(self.dropout_rate)
#         self.flatten = layers.Flatten()
#         self.classifier = layers.Dense(self.output_dim, name="classifier")
        
#         self.multihead_attn = None
#         self.ffn = None

#     def build(self, input_shape):
#         """
#         Creates layers with shapes dependent on the input data.
#         """
#         _, _, num_features = input_shape

#         self.multihead_attn = layers.MultiHeadAttention(
#             num_heads=self.num_heads,
#             key_dim=self.key_dim
#         )
#         self.ffn = keras.Sequential(
#             [
#                 layers.Dense(self.ffn_intermediate_dim, activation="relu"),
#                 layers.Dense(num_features),
#             ],
#             name="feed_forward_network"
#         )
#         super().build(input_shape)

#     def call(self, inputs, training=False):
#         """
#         Defines the forward pass of the EEGTransformer model.
#         """
#         # --- 1. Input Standardization REINSTATED ---
#         # The features from EEGNet are not raw signals and must be normalized.
#         mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
#         std = tf.math.reduce_std(inputs, axis=1, keepdims=True)
#         x = (inputs - mean) / (std + 1e-6) # Z-score normalization across the time dimension

#         # --- 2. Add Positional Encoding ---
#         x = self.pos_encoding_layer(x)

#         # --- 3. Transformer Encoder Block ---
#         attn_output = self.multihead_attn(query=x, value=x, key=x)
#         if training:
#             attn_output = self.dropout1(attn_output, training=training)
#         x = self.norm1(x + attn_output)

#         ffn_output = self.ffn(x)
#         if training:
#             ffn_output = self.dropout2(ffn_output, training=training)
#         x = self.norm2(x + ffn_output)

#         # --- 4. Final Classifier ---
#         x = self.flatten(x)
#         output = self.classifier(x)
#         return output

# EEGTransformer_Class.py (Corrected and Final Version)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    """
    This layer creates a fixed positional encoding matrix and adds it to the input.
    The encoding is created once and is not a trainable weight.
    """
    def __init__(self, max_timepoints=31, num_features=96, **kwargs): # Adapted for Block 2 features
        super().__init__(**kwargs)
        pos_encoding = self.build_positional_encoding(max_timepoints, num_features)
        self.positional_encoding = tf.Variable(pos_encoding, trainable=False)

    def build_positional_encoding(self, max_timepoints, num_features):
        angle_rates = 1 / (10000 ** ((2 * (tf.range(num_features, dtype=tf.float32) // 2)) / float(num_features)))
        positions = tf.range(max_timepoints, dtype=tf.float32)[:, tf.newaxis]
        angle_rads = positions * angle_rates[tf.newaxis, :]
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]

    def call(self, inputs):
        """Adds the positional encoding to the input tensor."""
        return inputs + self.positional_encoding

class EEGTransformer(keras.Model):
    """
    Final corrected TensorFlow/Keras model for EEG data using a Transformer.
    """
    def __init__(self, output_dim, num_heads, key_dim, ffn_intermediate_dim, dropout_rate=0.1, name='EEGTransformer'):
        super(EEGTransformer, self).__init__(name=name)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ffn_intermediate_dim = ffn_intermediate_dim
        self.dropout_rate = dropout_rate

        self.pos_encoding_layer = PositionalEncoding()
        self.multihead_attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = None 
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.flatten = layers.Flatten()
        self.classifier = layers.Dense(self.output_dim)

    def build(self, input_shape):
        _, _, num_features = input_shape
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ffn_intermediate_dim, activation="relu"),
                layers.Dense(num_features),
            ],
            name="feed_forward_network"
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.pos_encoding_layer(inputs)
        attn_output = self.multihead_attn(query=x, value=x, key=x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.norm2(x + ffn_output)
        x = self.flatten(x)
        output = self.classifier(x)
        return output