# %% [markdown]
# # 50.007 Machine Learning, Spring 2025
# # Design Project
# 
# Due 27 Apr 2025, 5:00pm
# 
# By: Aishwarya Iyer (1007141) and Khoo Zi Qi (1006984)

# %% [markdown]
# ## Part 1 (30points)

# %%
import numpy as np
import os
from collections import defaultdict
import math


# %%
train_file = "EN/train"
# print("File exists:", os.path.exists(train_file))

gold_file = "EN/dev.out"

dev_in_file = 'EN/dev.in'


# %% [markdown]
# Write a function that estimates the emission parameters from the training set using MLE (maximum likelihood estimation):

# %%
"""
Computes emission parameters for an HMM: e(x|y) = Count(y → x) / Count(y)
where:
- x: observed word
- y: corresponding tag (e.g., 'B-NP', 'I-VP', 'O')
"""
# Use defaultdict to automatically handles missing keys
from collections import defaultdict

def compute_emission_parameters(train_file_path):
    """
    Args:
        train_file_path: Path to training file (word-tag pairs separated by whitespace)
    
    Returns:
        Dictionary of dictionaries: emission_parameters[tag][word] = probability
    """
    
    # Initialize counters:
    # - emission_counts[tag][word] = times word appears with tag
    # - tag_counts[tag] = total occurrences of tag
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    # Count word-tag co-occurrences and tag frequencies
    with open(train_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    word, tag = line.split()  # Split by any whitespace
                    emission_counts[tag][word] += 1
                    tag_counts[tag] += 1
                except ValueError:
                    print(f"Skipping invalid line: {line}")

    # Calculate emission probabilities
    emission_parameters = defaultdict(dict)
    for tag in emission_counts:
        total_tag_occurrences = tag_counts[tag]
        for word in emission_counts[tag]:
            emission_parameters[tag][word] = (
                emission_counts[tag][word] / total_tag_occurrences
            )
    
    return emission_parameters

emission_parameters = compute_emission_parameters(train_file)
# print(emission_parameters)

# %% [markdown]
# Use smoothing
# - Identify words that appear less than 3 times
# - Replace those words with #UNK#
# 

# %%
def compute_emission_parameters_smoothing(train_file_path, k=3):

    """
    Args:
        train_file_path: Path to training file (word-tag pairs separated by whitespace)
        k: minimum count of word. If word count less than k, replace word with #UNK#.
    
    Returns:
        Dictionary of dictionaries: emission_parameters[tag][word] = probability
    """
    
    word_counts = defaultdict(int)
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # not an empty line
                word, tag = line.split()
                word_counts[word] += 1
    
    # Identify rare words
    rare_words = {word for word, count in word_counts.items() if count < k}

    # Initialize counters:
    # - emission_counts[tag][word] = times word appears with tag
    # - tag_counts[tag] = total occurrences of tag
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    # Count word-tag co-occurrences and tag frequencies
    with open(train_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    word, tag = line.split()  # Split by any whitespace
                    processed_word = word if word not in rare_words else '#UNK#' #modify training set
                    emission_counts[tag][processed_word] += 1
                    tag_counts[tag] += 1
                except ValueError:
                    print(f"Skipping invalid line: {line}")

    # Calculate emission probabilities
    emission_parameters = defaultdict(dict)
    for tag in emission_counts:
        total_tag_occurrences = tag_counts[tag]
        for word in emission_counts[tag]:
            emission_parameters[tag][word] = (
                emission_counts[tag][word] / total_tag_occurrences
            )
    
    return emission_parameters

emission_parameters_smoothing = compute_emission_parameters_smoothing(train_file, k = 3)
# print(emission_parameters_smoothing)

# %% [markdown]
# Implement a simple system that produces the tag
# y∗= arg maxy e(x|y)
# for each word x in the sequence.

# %%
def predict_tags(dev_in_file_path, emission_parameters, unknown_tag='O'):
    """
    Predicts tags for a sentence using emission probabilities.
    
    Args:
        sentence: List of words to tag
        emission_params: Dictionary from compute_emission_parameters()
        unknown_tag: Default tag for unseen words
    
    Returns:
        List of (word, predicted_tag) tuples
    """
    predicted = []
    
    with open(dev_in_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip()
            if word:  # Skip empty lines, each line has one word
                max_prob = -1
                best_tag = unknown_tag  # Default fallback
                try:
                    # Find tag with highest emission probability for this word
                    for tag in emission_parameters:
                        if word in emission_parameters[tag]:
                            if emission_parameters[tag][word] > max_prob:
                                max_prob = emission_parameters[tag][word]
                                best_tag = tag
                    
                    predicted.append((word, best_tag))
                except ValueError:
                    print(f"Skipping invalid line: {line}")         
    return predicted

predicted_list = predict_tags(dev_in_file, emission_parameters_smoothing)

# %% [markdown]
# Learn these parameters with train, and evaluate your system on the development set dev.in for
# each of the dataset. Write your output to dev.p2.out. (There's a typo in the project brief? Should be dev.p1.out)

# %%
def write_predictions(predicted_list, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as fout:
        for word, tag in predicted_list:
            fout.write(f"{word} {tag}\n")

write_predictions(predicted_list, 'EN/dev.p1.out')

# %% [markdown]
# Compare your outputs and the gold-standard outputs in dev.out and report the precision, recall and F scores of such a baseline system

# %%
# !python EvalScript/evalResult.py EN/dev.out EN/dev.p1.out


# %% [markdown]
# ## Part 2 (25 points)

# %%
def compute_transition_parameters(train_file_path, smoothing=0.1):
    """
    Computes transition probabilities with smoothing and proper STOP/START handling.
    
    Args:
        train_file_path: Path to training file (word-tag pairs separated by whitespace)
        smoothing: Laplace smoothing factor (default: 0.1)
    
    Returns:
        Dictionary of dictionaries: transition_parameters[prev_tag][tag] = probability
    """
    transition_counts = defaultdict(lambda: defaultdict(float))
    prev_tag_counts = defaultdict(float)
    all_tags = set()

    # Initialize with smoothing for all possible transitions
    with open(train_file_path, 'r', encoding='utf-8') as file:
        prev_tag = 'START'
        for line in file:
            line = line.strip()
            if line:  # Non-empty line (word-tag pair)
                try:
                    _, tag = line.split()
                    transition_counts[prev_tag][tag] += 1
                    prev_tag_counts[prev_tag] += 1
                    all_tags.add(tag)
                    prev_tag = tag
                except ValueError:
                    print(f"Skipping invalid line: {line}")
            else:  # Empty line (sentence boundary)
                transition_counts[prev_tag]['STOP'] += 1
                prev_tag_counts[prev_tag] += 1
                prev_tag = 'START'  # Reset for next sentence
        all_tags.add('STOP')

    # Apply Laplace smoothing and normalize
    transition_parameters = defaultdict(dict)
    for prev_tag in transition_counts:
        total = prev_tag_counts[prev_tag] + smoothing * len(all_tags)
        for tag in all_tags:
            count = transition_counts[prev_tag].get(tag, 0) + smoothing
            transition_parameters[prev_tag][tag] = count / total

    # Ensure START -> first tag is properly initialized
    transition_parameters['START'] = {
        tag: transition_counts['START'].get(tag, 0) / prev_tag_counts['START']
        for tag in all_tags if tag != 'STOP'
    }

    return transition_parameters

transition_parameters = compute_transition_parameters(train_file)
# print(transition_parameters)

# %%

def viterbi(sentence, transition_params, emission_params, all_tags):
    """
    Viterbi algorithm with robust probability handling
    """
    n = len(sentence)
    viterbi_matrix = defaultdict(dict)
    backpointer = defaultdict(dict)
    
    # Small epsilon value to avoid log(0)
    EPSILON = 1e-10
    
    # Initialize first step
    for tag in all_tags:
        # Handle emission probability
        emission_prob = emission_params[tag].get(sentence[0], EPSILON)
        
        # Handle transition probability
        trans_prob = transition_params['START'].get(tag, EPSILON)
        
        # Calculate log probabilities safely
        if emission_prob <= 0:
            emission_prob = EPSILON
        if trans_prob <= 0:
            trans_prob = EPSILON
            
        viterbi_matrix[0][tag] = math.log(trans_prob) + math.log(emission_prob)
        backpointer[0][tag] = 'START'
    
    # Recursion
    for t in range(1, n):
        word = sentence[t]
        for current_tag in all_tags:
            max_prob = -float('inf')
            best_prev_tag = None
            
            for prev_tag in all_tags:
                # Get probabilities safely
                trans_prob = transition_params[prev_tag].get(current_tag, EPSILON)
                emission_prob = emission_params[current_tag].get(word, EPSILON)
                
                if trans_prob <= 0:
                    trans_prob = EPSILON
                if emission_prob <= 0:
                    emission_prob = EPSILON
                
                current_prob = (viterbi_matrix[t-1][prev_tag] + 
                               math.log(trans_prob) + 
                               math.log(emission_prob))
                
                if current_prob > max_prob:
                    max_prob = current_prob
                    best_prev_tag = prev_tag
            
            viterbi_matrix[t][current_tag] = max_prob
            backpointer[t][current_tag] = best_prev_tag
    
    # Termination
    max_prob = -float('inf')
    best_last_tag = None
    for tag in all_tags:
        stop_prob = viterbi_matrix[n-1][tag] + math.log(transition_params[tag].get('STOP', EPSILON))
        if stop_prob > max_prob:
            max_prob = stop_prob
            best_last_tag = tag
    
    # Backtrace
    tags = [best_last_tag]
    for t in range(n-1, 0, -1):
        tags.append(backpointer[t][tags[-1]])
    tags.reverse()
    
    return tags

def run_viterbi_on_dev_set(dev_in_path, output_path, transition_params, emission_params, all_tags):
    """
    Runs Viterbi on development set and writes predictions
    """
    with open(dev_in_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        current_sentence = []
        for line in fin:
            line = line.strip()
            if line:
                current_sentence.append(line)
            else:
                if current_sentence:
                    predicted_tags = viterbi(current_sentence, transition_params, 
                                           emission_params, all_tags)
                    for word, tag in zip(current_sentence, predicted_tags):
                        fout.write(f"{word} {tag}\n")
                    fout.write("\n")
                current_sentence = []
        
        # Handle last sentence if file doesn't end with newline
        if current_sentence:
            predicted_tags = viterbi(current_sentence, transition_params, 
                                   emission_params, all_tags)
            for word, tag in zip(current_sentence, predicted_tags):
                fout.write(f"{word} {tag}\n")



# %%

emission_params = compute_emission_parameters_smoothing(train_file)
transition_params = compute_transition_parameters(train_file)
    

all_tags = set()
with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                _, tag = line.split()
                all_tags.add(tag)
            except ValueError:
                continue
    
# Run Viterbi on dev set
run_viterbi_on_dev_set(dev_in_file, "EN/dev.p2.out", transition_params, emission_params, all_tags)
    
# Evaluate
# metrics = evaluate(gold_file, "EN/dev.p2.out")
# print_metrics(metrics)

# %%
# !python EvalScript/evalResult.py EN/dev.out EN/dev.p2.out

# %% [markdown]
# ## Part 3 (25 points)

# %%
def kth_best_viterbi(sentence, transition_params, emission_params, all_tags, k=4):
    """
    Modified Viterbi algorithm to find the k-th best sequence
    """
    n = len(sentence)
    viterbi_matrix = defaultdict(dict)  # viterbi_matrix[t][tag] = list of top k probabilities
    backpointer = defaultdict(dict)     # backpointer[t][tag] = list of top k previous tags
    
    EPSILON = 1e-10
    
    # Initialize first step (Base case)
    for tag in all_tags:
        emission_prob = emission_params[tag].get(sentence[0], EPSILON)
        trans_prob = transition_params['START'].get(tag, EPSILON)
        
        if emission_prob <= 0:
            emission_prob = EPSILON
        if trans_prob <= 0:
            trans_prob = EPSILON
            
        prob = math.log(trans_prob) + math.log(emission_prob)
        viterbi_matrix[0][tag] = [prob]
        backpointer[0][tag] = [('START', 0)]  # (prev_tag, path_index)
    
    # Recursion
    for t in range(1, n):
        word = sentence[t]
        for current_tag in all_tags:
            all_paths = []
            
            for prev_tag in all_tags:
                if prev_tag not in viterbi_matrix[t-1]:
                    continue
                    
                trans_prob = transition_params[prev_tag].get(current_tag, EPSILON)
                emission_prob = emission_params[current_tag].get(word, EPSILON)
                
                if trans_prob <= 0:
                    trans_prob = EPSILON
                if emission_prob <= 0:
                    emission_prob = EPSILON
                
                # For each of the top k paths to prev_tag
                for path_idx, prev_prob in enumerate(viterbi_matrix[t-1][prev_tag]):
                    current_prob = prev_prob + math.log(trans_prob) + math.log(emission_prob)
                    all_paths.append((current_prob, prev_tag, path_idx))
            
            # Sort all paths and keep top k
            all_paths.sort(reverse=True, key=lambda x: x[0])
            top_k_paths = all_paths[:k]
            
            if top_k_paths:
                viterbi_matrix[t][current_tag] = [prob for prob, _, _ in top_k_paths]
                backpointer[t][current_tag] = [(prev_tag, path_idx) for _, prev_tag, path_idx in top_k_paths]
    
    # Termination - find top k paths ending with STOP
    final_paths = []
    for tag in all_tags:
        if tag not in viterbi_matrix[n-1]:
            continue
            
        stop_prob = math.log(transition_params[tag].get('STOP', EPSILON))
        for path_idx, prob in enumerate(viterbi_matrix[n-1][tag]):
            final_prob = prob + stop_prob
            final_paths.append((final_prob, tag, path_idx))
    
    final_paths.sort(reverse=True, key=lambda x: x[0])
    
    # If there are fewer than k paths, return the last one
    if len(final_paths) < k:
        k = len(final_paths)
    
    # Get the k-th best path (0-indexed, so k=3 for 4th best)
    if k == 0:
        return []
    
    selected_path = final_paths[k-1]
    final_tag, path_idx = selected_path[1], selected_path[2]
    
    # Backtrace
    tags = [final_tag]
    for t in range(n-1, 0, -1):
        prev_tag, prev_path_idx = backpointer[t][tags[-1]][path_idx]
        tags.append(prev_tag)
        path_idx = prev_path_idx
    
    tags.reverse()
    return tags

def run_kth_best_viterbi(dev_in_path, output_path, transition_params, emission_params, all_tags, k=4):
    """
    Runs k-th best Viterbi on development set and writes predictions
    """
    with open(dev_in_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        current_sentence = []
        for line in fin:
            line = line.strip()
            if line:
                current_sentence.append(line)
            else:
                if current_sentence:
                    predicted_tags = kth_best_viterbi(current_sentence, transition_params, 
                                                    emission_params, all_tags, k)
                    for word, tag in zip(current_sentence, predicted_tags):
                        fout.write(f"{word} {tag}\n")
                    fout.write("\n")
                current_sentence = []
        
        # Handle last sentence if file doesn't end with newline
        if current_sentence:
            predicted_tags = kth_best_viterbi(current_sentence, transition_params, 
                                            emission_params, all_tags, k)
            for word, tag in zip(current_sentence, predicted_tags):
                fout.write(f"{word} {tag}\n")

# change k for k best variation
run_kth_best_viterbi(dev_in_file, "EN/dev.p3.out", transition_params, emission_params, all_tags, k=4)

# %%
# !python EvalScript/evalResult.py EN/dev.out EN/dev.p3.out



