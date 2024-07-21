import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array


def preprocess_image(image_path, target_size=(225, 225)):
    """
    Preprocesses an image for use in the model.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple, optional): Target size for resizing the image.
            Defaults to (225, 225).

    Returns:
        np.ndarray: The preprocessed image as a NumPy array.
    """
    img = Image.open(image_path)
    x = img_to_array(img)
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def create_data_generators(
    train_dir, test_dir, target_size=(225, 225), batch_size=32, class_mode="categorical"
):
    """
    Creates training and validation data generators using ImageDataGenerator.

    Args:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        target_size (tuple, optional): Target size for resizing images.
            Defaults to (225, 225).
        batch_size (int, optional): Batch size for the data generators.
            Defaults to 32.
        class_mode (str, optional): Class mode for the data generators.
            Defaults to 'categorical'.

    Returns:
        tuple: A tuple containing the training and validation data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode=class_mode
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir, target_size=target_size, batch_size=batch_size, class_mode=class_mode
    )

    return train_generator, validation_generator


def predict_on_image(model, image_path, target_size=(225, 225)):
    """
    Predicts the class label for a given image.

    Args:
        model (tf.keras.Model): The trained CNN model.
        image_path (str): Path to the image file.
        target_size (tuple, optional): Target size for resizing the image.
            Defaults to (225, 225).

    Returns:
        tuple: A tuple containing the predicted class label and its
        probability.
    """
    preprocessed_image = preprocess_image(image_path, target_size)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    predicted_probability = prediction[0][predicted_class]

    return predicted_class, predicted_probability
