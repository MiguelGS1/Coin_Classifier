import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config import IMAGE_SIZE, LEARNING_RATE


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="image")

    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = layers.RandomFlip("horizontal", name="aug_flip")(x)
    x = layers.RandomRotation(0.05, name="aug_rotation")(x)
    x = layers.RandomZoom(0.08, name="aug_zoom")(x)
    x = layers.RandomContrast(0.08, name="aug_contrast")(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.35)(x)
    embedding = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="embedding",
    )(x)
    logits = layers.Dense(num_classes, activation="softmax", name="class_probs")(embedding)

    model = models.Model(inputs=inputs, outputs=logits, name="coin_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



def build_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output,
        name="coin_feature_extractor",
    )
