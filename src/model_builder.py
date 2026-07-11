# -*- coding: utf-8 -*-

"""
Configurable CNN and ResNet50 model construction.

"""

import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)

from config import IMAGE_SIZE


@tf.keras.utils.register_keras_serializable(package="ProjetPratique") # Registers the custom layer so the model can be serialized.
class PretraitementResNet50(tf.keras.layers.Layer):
    """
    Applies the official ResNet50 preprocessing.

    This serializable layer can be saved and reloaded with the model.
    """

    def call(self, inputs):
        return preprocess_input(inputs)

    def get_config(self):
        return super().get_config()


def construire_augmentation(config):
    """Builds the image augmentation pipeline."""

    layers = []

    if config.get("random_flip", True):
        layers.append(
            RandomFlip(
                mode="horizontal",
                name="random_flip",
            )
        )

    rotation = config.get("random_rotation", 0.10)

    if rotation > 0:
        layers.append(
            RandomRotation(
                factor=rotation,
                name="random_rotation",
            )
        )

    zoom = config.get("random_zoom", 0.10)

    if zoom > 0:
        layers.append(
            RandomZoom(
                height_factor=zoom,
                width_factor=zoom,
                name="random_zoom",
            )
        )

    contraste = config.get("random_contrast", 0.10)

    if contraste > 0:
        layers.append(
            RandomContrast(
                factor=contraste,
                name="random_contrast",
            )
        )

    return Sequential(
        layers,
        name="augmentation_donnees",
    )


def creer_optimiseur(optimizer_name, learning_rate):
    """Creates the requested optimizer."""

    optimizer_name = optimizer_name.lower().strip()

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate
        )

    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.90,
        )

    if optimizer_name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

    raise ValueError(
        f"Unknown optimizer : '{optimizer_name}'."
    )


def compiler_modele(model, config):
    """Compiles the classification model."""

    optimizer = creer_optimiseur(
        config.get("optimizer", "adam"),
        config.get("learning_rate", 1e-4),
    )

    model.compile(
        optimizer=optimizer,
        loss=config.get(
            "loss",
            "sparse_categorical_crossentropy",
        ),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


def construire_cnn(number_of_classes, config):
    """Builds a configurable CNN."""

    inputs = Input(
        shape=(*IMAGE_SIZE, 3),
        name="image_entree",
    )

    x = construire_augmentation(config)(inputs)

    x = Rescaling(
        scale=1.0 / 255.0,
        name="normalisation",
    )(x)

    for index, filters in enumerate(
        config.get(
            "conv_filters",
            [32, 64, 128, 256],
        ),
        start=1,
    ):
        x = Conv2D(
            filters=filters,
            kernel_size=config.get(
                "kernel_size",
                3,
            ),
            padding=config.get(
                "padding",
                "same",
            ),
            activation=config.get(
                "activation",
                "relu",
            ),
            name=f"conv_{index}",
        )(x)

        if config.get(
            "use_batch_normalization",
            True,
        ):
            x = BatchNormalization(
                name=f"batch_norm_{index}"
            )(x)

        x = MaxPooling2D(
            pool_size=config.get(
                "pool_size",
                2,
            ),
            name=f"pool_{index}",
        )(x)

    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    dense_units = config.get(
        "dense_units",
        256,
    )

    if dense_units > 0:
        x = Dense(
            units=dense_units,
            activation=config.get(
                "activation",
                "relu",
            ),
            name="dense_classification",
        )(x)

    dropout_rate = config.get(
        "dropout_rate",
        0.50,
    )

    if dropout_rate > 0:
        x = Dropout(
            rate=dropout_rate,
            name="dropout_classification",
        )(x)

    outputs = Dense(
        units=number_of_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_configurable",
    )

    return compiler_modele(
        model,
        config,
    )


def construire_resnet50(number_of_classes, config):
    """Builds a configurable ResNet50 model."""

    inputs = Input(
        shape=(*IMAGE_SIZE, 3),
        name="image_entree",
    )

    x = construire_augmentation(config)(inputs)

    x = PretraitementResNet50(
        name="pretraitement_resnet"
    )(x)

    base_model = ResNet50(
        weights=config.get(
            "weights",
            "imagenet",
        ),
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )

    base_model.trainable = config.get(
        "trainable_base",
        False,
    )

    x = base_model(
        x,
        training=False,
    )

    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    dense_units = config.get(
        "dense_units",
        256,
    )

    if dense_units > 0:
        x = Dense(
            units=dense_units,
            activation=config.get(
                "activation",
                "relu",
            ),
            name="dense_classification",
        )(x)

    dropout_rate = config.get(
        "dropout_rate",
        0.40,
    )

    if dropout_rate > 0:
        x = Dropout(
            rate=dropout_rate,
            name="dropout_classification",
        )(x)

    outputs = Dense(
        units=number_of_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="resnet50_configurable",
    )

    return compiler_modele(
        model,
        config,
    )


def build_model(
    model_name,
    number_of_classes,
    model_config,
):
    """Builds the selected model."""

    normalized_name = model_name.lower().strip()

    if normalized_name == "cnn":
        return construire_cnn(
            number_of_classes,
            model_config,
        )

    if normalized_name == "resnet50":
        return construire_resnet50(
            number_of_classes,
            model_config,
        )

    raise ValueError(
        f"Unknown model : '{model_name}'."
    )