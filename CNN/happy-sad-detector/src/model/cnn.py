from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    input_shape: tuple[int, int, int] = (256, 256, 3),
) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),

            layers.Conv2D(16, (3, 3), strides=1, activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(32, (3, 3), strides=1, activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(16, (3, 3), strides=1, activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],

        name="happy_sad_cnn",
    )

    return model
