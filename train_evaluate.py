from matplotlib import pyplot as plt
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_preprocessing import create_data_generators
from model import create_model


def train_model(train_dir, valid_dir, epochs=5, batch_size=32):
    """
    Trains the CNN model using the provided data generators.

    Args:
        train_dir (str): Path to the training data directory.
        valid_dir (str): Path to the validation data directory.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        batch_size (int, optional): Batch size for training. Defaults to 32.

    Returns:
        history: The training history object containing accuracy and loss
        metrics.
    """
    train_generator, validation_generator = create_data_generators(train_dir, valid_dir)
    model = create_model()

    history = model.fit(
        train_generator,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_batch_size=batch_size,
    )

    # Plot training and validation loss/accuracy
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()

    return history


def evaluate_model(model, test_dir, target_size=(225, 225)):
    """
    Evaluates the trained model on the test data using multiple metrics.

    Args:
        model (tf.keras.Model): The trained CNN model.
        test_dir (str): Path to the test data directory.
        target_size (tuple, optional): Target size for resizing test images.
            Defaults to (225, 225).

    Returns:
        None: Prints the model's evaluation metrics on the test set.
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=target_size, class_mode="categorical"
    )

    # Evaluate and calculate multiple metrics
    loss, accuracy = model.evaluate(test_generator)
    precision = Precision(name="precision")
    recall = Recall(name="recall")
    auc = AUC(name="auc")
    model.compile(loss="categorical_crossentropy", metrics=[precision, recall, auc])
    _, test_precision, test_recall, test_auc = model.evaluate(test_generator)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"Test set Loss: {loss:.4f}")
    print(f"Test set Accuracy: {accuracy:.4f}")
    print(f"Test set Precision: {test_precision:.4f}")
    print(f"Test set Recall: {test_recall:.4f}")
    print(f"Test set AUC: {test_auc:.4f}")


if __name__ == "__main__":
    # Define data paths (modify as needed)
    train_dir = "Dataset/Train/Train"
    valid_dir = "Dataset/Validation/Validation"
    test_dir = "Dataset/Test/Test"

    # Train the model
    history = train_model(train_dir, valid_dir)

    # Evaluate the model on the test set
    evaluate_model(history.model, test_dir)

    # Save the model (optional)
    history.model.save("models/model.h5")
