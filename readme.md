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

###  Running the Project

This project is implemented using Jupyter Notebooks. To run it:

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the notebook files.
3. Run all cells from top to bottom.

Alternatively, you can use Visual Studio Code (VSCode) to run the Jupyter Notebooks.

### Files

- `part_1_2_3.ipynb`: Contains Parts 1 to 3:
  - **Part 1**: Estimates emission parameters from the training set using Maximum Likelihood Estimation (MLE), both unsmoothed and smoothed. Implements a simple system that computes the most likely tag \( y^* = \arg\max_{y} e(x|y) \) for each word \( x \) in the sequence.
  - **Part 2**: Computes transition probabilities with smoothing and proper handling of STOP/START tokens. Implements the Viterbi algorithm.
  - **Part 3**: A modified version of the Viterbi algorithm to find the k-th best sequence.

- `part_4.ipynb`: Implements the Averaged Perceptron model.

