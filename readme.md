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

## Part 1 Function Descriptions

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

## Part 2 Function Descriptions

