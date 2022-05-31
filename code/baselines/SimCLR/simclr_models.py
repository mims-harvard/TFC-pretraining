import tensorflow as tf

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2020 C. I. Tang"

"""
Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

def create_base_model(input_shape, model_name="base_model"):
    """
    Create the base model for activity recognition
    Reference (TPN model):
        Saeed, A., Ozcelebi, T., & Lukkien, J. (2019). Multi-task self-supervised learning for human activity detection. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(2), 1-30.

    Architecture:
        Input
        -> Conv 1D: 32 filters, 24 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 64 filters, 16 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 96 filters, 8 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Global Maximum Pooling 1D
    
    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
    
    Returns:
        model (tf.keras.Model)
    """

    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.Conv1D(
            32, 24,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(
            64, 16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Conv1D(
        96, 8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    return tf.keras.Model(inputs, x, name=model_name)

def attach_simclr_head(base_model, hidden_1=256, hidden_2=128, hidden_3=50):
    """
    Attach a 3-layer fully-connected encoding head

    Architecture:
        base_model
        -> Dense: hidden_1 units
        -> ReLU
        -> Dense: hidden_2 units
        -> ReLU
        -> Dense: hidden_3 units
    """

    input = base_model.input
    x = base_model.output

    projection_1 = tf.keras.layers.Dense(hidden_1)(x)
    projection_1 = tf.keras.layers.Activation("relu")(projection_1)
    projection_2 = tf.keras.layers.Dense(hidden_2)(projection_1)
    projection_2 = tf.keras.layers.Activation("relu")(projection_2)
    projection_3 = tf.keras.layers.Dense(hidden_3)(projection_2)

    simclr_model = tf.keras.Model(input, projection_3, name= base_model.name + "_simclr")

    return simclr_model


def create_linear_model_from_base_model(base_model, output_shape, intermediate_layer=7):

    """
    Create a linear classification model from the base mode, using activitations from an intermediate layer

    Architecture:
        base_model-intermediate_layer
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: SGD
    Loss: CategoricalCrossentropy

    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories

        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    inputs = base_model.inputs
    x = base_model.layers[intermediate_layer].output
    x = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.random_normal_initializer(stddev=.01))(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=base_model.name + "linear")

    for layer in model.layers[:intermediate_layer+1]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model


def create_full_classification_model_from_base_model(base_model, output_shape, model_name="TPN", intermediate_layer=7, last_freeze_layer=4):
    """
    Create a full 2-layer classification model from the base mode, using activitations from an intermediate layer with partial freezing

    Architecture:
        base_model-intermediate_layer
        -> Dense: 1024 units
        -> ReLU
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: Adam
    Loss: CategoricalCrossentropy

    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories

        model_name
            name of the output model

        intermediate_layer
            the index of the intermediate layer from which the activations are extracted

        last_freeze_layer
            the index of the last layer to be frozen for fine-tuning (including the layer with the index)
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    # inputs = base_model.inputs
    intermediate_x = base_model.layers[intermediate_layer].output

    x = tf.keras.layers.Dense(1024, activation='relu')(intermediate_x)
    x = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name=model_name)

    for layer in model.layers:
        layer.trainable = False
    
    for layer in model.layers[last_freeze_layer+1:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    """
    Create an intermediate model from base mode, which outputs embeddings of the intermediate layer

    Parameters:
        base_model
            the base model from which the intermediate model is built
        
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted

    Returns:
        model (tf.keras.Model)
    """

    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model

