import os

from utils import total_files

# Define data paths (modify as needed)
train_path = "Dataset/Train/Train"
test_path = "Dataset/Test/Test"
valid_path = "Dataset/Validation/Validation"

# Define disease class subfolders (modify as needed)
disease_classes = ["Healthy", "Powdery", "Rust"]


def print_file_counts(path, disease_classes):
    """
    Prints the number of images for each disease class in a given data path.

    Args:
        path (str): Path to the data directory (train, test, or validation).
        disease_classes (list): List of disease class names.
    """
    for disease_class in disease_classes:
        class_path = os.path.join(path, disease_class)
        num_files = total_files(class_path)
        print(f"Number of {disease_class} leaf images: {num_files}")


if __name__ == "__main__":
    # Print file counts for train, test, and validation sets
    print("Train set:")
    print_file_counts(train_path, disease_classes)
    print("\nTest set:")
    print_file_counts(test_path, disease_classes)
    print("\nValidation set:")
    print_file_counts(valid_path, disease_classes)
