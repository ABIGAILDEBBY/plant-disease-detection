from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def create_model(input_shape=(225, 225, 3)):
    """
    Defines the CNN model architecture.

    Args:
        input_shape (tuple, optional): Input shape for the model.
            Defaults to (225, 225, 3) for color images with specified
            dimensions.

    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation="softmax"))  # Assuming 3 disease classes

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
