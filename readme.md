# 50.007 Machine Learning, Spring 2025
# Design Project README

By: Aishwarya Iyer	1007141, Khoo Zi Qi	1006984
## Overview

This project implements key components of a Hidden Markov Model (HMM) for Named Entity Recognition (NER). It includes functions to compute emission parameters, apply smoothing for rare words, generate predictions, and evaluate model performance using precision, recall, and F1 score.

The code assumes a training dataset in which each line contains a word and its corresponding tag, separated by whitespace. Sentences are separated by blank lines.


##  Setup Instructions

###  Dependencies
Make sure you have Python installed. Then install required packages:

```bash
pip install numpy dill
```
Note: dill is only required if you want to save/load the model in part 4.

### Running the Project

This project is implemented in Python.

1. **Navigate to the root folder:**
   Open the command line and change directory to the project root folder, `ml_project`.

2. **To run Parts 1 to 3:**
   ```bash
   python .\part_1_2_3.py
   ```

3. **To run Part 4:**
   ```bash
   python .\part_4.py
   ```
The model is saved and loaded using `dill` to/from the file `part_4_model.pkl`, so you don't need to retrain it.

4. **To run the evaluation script (e.g., for Part 1):**
   ```bash
   python EvalScript/evalResult.py EN/dev.out EN/dev.p1.out
   ```

### Files

- **`part_1_2_3.ipynb`:** Contains Parts 1 to 3:
  - **Part 1:** Estimates emission parameters from the training set using Maximum Likelihood Estimation (MLE), both unsmoothed and smoothed. Implements a simple system to compute the most likely tag \( y^* = \arg\max_{y} e(x|y) \) for each word \( x \) in the sequence.
  - **Part 2:** Computes transition probabilities with smoothing and proper handling of STOP/START tokens. Implements the Viterbi algorithm.
  - **Part 3:** A modified version of the Viterbi algorithm to find the k-th best sequence.

- **`part_4.ipynb`:** Implements the Averaged Perceptron model.
- **`eval_log.txt`:** Contains evaluation output for Part 4.
- **`EN/`:** Contains the following output files:
  1. `dev.p1.out`
  2. `dev.p2.out`
  3. `dev.p3.out`
  4. `dev.p4.out`
  5. `test.p4.out`


