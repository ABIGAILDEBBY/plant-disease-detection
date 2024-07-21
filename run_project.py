from data_exploration import print_file_counts
from train_evaluate import evaluate_model, train_model

# Define data paths (modify as needed)
train_dir = "Dataset/Train/Train"
valid_dir = "Dataset/Validation/Validation"
test_dir = "Dataset/Test/Test"

# Explore data distribution (optional)
print_file_counts(train_dir, ["Healthy", "Powdery", "Rust"])
# Modify disease classes if needed

# Train the model
history = train_model(train_dir, valid_dir)

# Evaluate the model
evaluate_model(history.model, test_dir)
