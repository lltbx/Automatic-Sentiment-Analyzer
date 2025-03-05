# Automatic Sentiment Analyzer

![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

Automatic Sentiment Analyzer is designed to implement and evaluate linear classifiers for sentiment analysis. This project focuses on the Perceptron algorithm and its variations, utilizing loss functions such as the hinge-loss function. It includes a feature matrix where rows are feature vectors, and columns are individual features, along with a vector of labels representing the actual sentiment of the corresponding features.

## Features

- **Linear Classifiers**: Implementations of Perceptron, average Perceptron, and Pegasos algorithms.
- **Loss Functions**: Includes hinge-loss and other loss functions.
- **Feature Extraction**: Uses bag-of-words model for feature extraction.
- **Hyperparameter Tuning**: Capabilities for tuning hyperparameters to optimize model performance.
- **Accuracy Evaluation**: Mechanisms to evaluate training, validation, and test accuracy.


### Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning the repository)

### Installation
   Bash:
   git clone https://github.com/lltbx/Automatic-Sentiment-Analyzer.git
   cd Automatic-Sentiment-Analyzer

### Data Loading

The project loads training, validation, and test data from `.tsv` files. Ensure the data files are in the correct format.

```python
import utils

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')
```

### Feature Extraction

Use the `bag_of_words` method to create a dictionary and extract bag-of-words feature vectors for training, validation, and test data.

```python
import project1 as p1

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
dictionary = p1.bag_of_words(train_texts)
train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
```

### Training and Evaluation

Train the classifiers and evaluate their performance.

```python
# Example: Training Pegasos algorithm
theta, theta_0 = p1.pegasos(train_bow_features, train_labels, T=25, L=0.01)

# Make predictions on the test set
test_predictions = p1.classify(test_bow_features, theta, theta_0)
test_accuracy = p1.accuracy(test_predictions, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### Hyperparameter Tuning

Tune hyperparameters like T (iterations) and L (lambda regularization) for optimal performance.

```python
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]
pct_tune_results = utils.tune_perceptron(Ts, train_bow_features, train_labels, val_bow_features, val_labels)
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
```