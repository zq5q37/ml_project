# 50.007 Machine Learning, Spring 2025
# Design Project Report

## Overview

This project implements key components of a Hidden Markov Model (HMM) for Named Entity Recognition (NER). It includes functions to compute emission parameters, apply smoothing for rare words, generate predictions, and evaluate model performance using precision, recall, and F1 score.

The code assumes a training dataset in which each line contains a word and its corresponding tag, separated by whitespace. Sentences are separated by blank lines.

---

## Setup Instructions

### Dependencies
Ensure you have python.
Ensure `numpy` is installed in your environment. If not, install it using:

```bash
pip install numpy
```

### Running the Code
The implementation is contained in a Jupyter Notebook. To run the project:

1. Open the notebook.
2. Execute all cells sequentially from top to bottom.

This will compute the emission probabilities, apply smoothing, optionally write predictions to a file, and print evaluation metrics.

---

## Part 1

### `compute_emission_parameters(train_file_path)`
Calculates unsmoothed emission probabilities.

- **Input:** Path to training data file.
- **Output:** A nested dictionary `emission_parameters[tag][word] = probability`.

---

### `compute_emission_parameters_smoothing(train_file_path, k)`
Adds smoothing to handle rare words.

- Words with frequency < `k` are replaced with a special token `#UNK#`.
- The emission probabilities are then computed using this modified training data.

- **Input:** Training file path, threshold `k`.
- **Output:** Smoothed emission parameters dictionary.

---

### `write_predictions(predicted_list, output_file_path)`
Writes predicted `(word, tag)` pairs to a file, one pair per line.

- **Input:** A list of predicted pairs and an output file path.

---

### `extract_chunks(tag_sequence)`
Converts a sequence of tags into a list of spans representing entities.

- **Input:** List of tags (`['B-LOC', 'I-LOC', 'O', ...]`)
- **Output:** List of tuples `(start_idx, end_idx, chunk_type)`

---

### `evaluate(gold_file, pred_file)`
Evaluates predicted tags against the gold standard.

- **Input:** File paths to the gold and predicted tag sequences.
- **Output:** A dictionary containing precision, recall, F1 score, and counts for TP, FP, FN.

---

### `print_metrics(metrics)`
Formats and prints the results from the evaluation function.

---

## Part 2

---
## Part 3

---
## Part 4

### Model and Method: Averaged Perceptron for Sequence Labeling

We implemented a **first-order sequence labeling model** using the **Averaged Perceptron** algorithm. This model is used to assign chunk-level tags (e.g., `B-NP`, `I-VP`, `O`) to each token in a sentence, learning from features that capture both the word and its context.

---

### Model Overview

The **Averaged Perceptron** is an online, linear classifier. It updates its weights based on prediction errors and averages weights across all iterations to improve generalization and reduce overfitting. At each step, it:

1. Predicts a label using current feature weights.
2. Compares the prediction with the gold label.
3. If incorrect, updates the weights for the correct and incorrect labels.
4. Averages weights over time.

The update rule for label weights is:

$$
w_{y} = w_{y} + \phi(x), \quad w_{\hat{y}} = w_{\hat{y}} - \phi(x)
$$


Averaged weights are calculated as:

$$
\bar{w} = \frac{1}{T} \sum_{t=1}^{T} w^{(t)}
$$

where \( T \) is the total number of updates.

---

### Implementation Details

#### `AveragedPerceptron` Class

This class implements the core algorithm:

```python
class AveragedPerceptron:
    def __init__(self):
        self.weights = defaultdict(lambda: defaultdict(float))
        self.totals = defaultdict(lambda: defaultdict(float))
        self.timestamps = defaultdict(lambda: defaultdict(int))
        self.i = 0  # Global update counter

    def predict(self, features):
        scores = defaultdict(float)
        for feat, value in features.items():
            for label, weight in self.weights[feat].items():
                scores[label] += value * weight
        return max(scores, key=scores.get) if scores else 'O'

    def update(self, truth, guess, features):
        self.i += 1
        if truth == guess:
            return
        for feat, value in features.items():
            self._update_feat(feat, truth, value)
            self._update_feat(feat, guess, -value)

    def _update_feat(self, feat, label, value):
        weight = self.weights[feat][label]
        self.totals[feat][label] += (self.i - self.timestamps[feat][label]) * weight
        self.timestamps[feat][label] = self.i
        self.weights[feat][label] += value

    def average_weights(self):
        for feat, weights in self.weights.items():
            for label in weights:
                total = self.totals[feat][label]
                total += (self.i - self.timestamps[feat][label]) * weights[label]
                averaged = total / self.i
                if averaged:
                    self.weights[feat][label] = averaged
                else:
                    del self.weights[feat][label]
```

---

### Feature Extraction

#### `extract_features(sentence, i)`

Generates a dictionary of features for the word at position `i`. These features include:

- Current word
- Previous and next word
- Prefix and suffix
- Capitalization, digit, title-case status

```python
def extract_features(sentence, i):
    word = sentence[i]
    features = {
        'w': word,
        'w-1': sentence[i-1] if i > 0 else '<START>',
        'w+1': sentence[i+1] if i < len(sentence) - 1 else '<END>',
        'prefix': word[:3],
        'suffix': word[-3:],
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit()
    }
    return features
```

These features provide context-aware information for sequence tagging.

---

### Training Procedure

#### `train_perceptron(dataset, dev_data, gold_dev_tags, epochs=5)`

Trains the model over multiple epochs and optionally evaluates on development data each time.

```python
def train_perceptron(dataset, dev_data, gold_dev_tags, epochs=5):
    model = AveragedPerceptron()
    for epoch in range(epochs):
        for sentence, tags in dataset:
            for i in range(len(sentence)):
                features = extract_features(sentence, i)
                pred = model.predict(features)
                model.update(tags[i], pred, features)
        model.average_weights()
    return model
```

---

### Evaluation

Evaluation is based on **chunk-level F1 score** using EvalScript/evalResult.py. A predicted chunk is correct if its label and span match the gold standard. Metrics computed:

- **Precision** = Correct chunks / Predicted chunks
- **Recall** = Correct chunks / Gold chunks
- **F Score** = 2 × (Precision × Recall) / (Precision + Recall)

---

