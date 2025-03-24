import os
from src.utils import load_dataset
from src.id3 import build_tree

# Define dataset path and parameters
dataset_path = os.path.join("data", "diabetes_dataset.csv")
data = load_dataset(dataset_path)
target = "diabet"


train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
print(
    f"Splitting the dataset into training and testing subsets (70% training, 30% testing)."
)
train_data = data[:train_size]
test_data = data[train_size:]



def classify(sample, node):

    if node.is_leaf():
        return node.label
    if sample[node.column] == node.value:
        return classify(sample, node.true_branch)
    else:
        return classify(sample, node.false_branch)


decision_tree = build_tree(train_data, target)
print("Training complete! Decision tree is built.")
correct_train = 0
for _, row in train_data.iterrows():
    prediction = classify(row, decision_tree)
    actual = row[target]
    if prediction == actual:
        correct_train += 1

train_accuracy = correct_train / len(train_data) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")


correct_predictions = 0  # For accuracy calculation
total_samples = len(test_data)

print("\nPredictions on test data:")
for index, row in test_data.iterrows():
    prediction = classify(row, decision_tree)  # Predict using the decision tree
    actual = row[target]  # True label
    print(f"Sample {index + 1}: Prediction = {prediction}, Actual = {actual}")

    # Count correct predictions
    if prediction == actual:
        correct_predictions += 1

accuracy = correct_predictions / total_samples * 100
print(f"\nAccuracy on test data: {accuracy:.2f}%")
