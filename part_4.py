# %% [markdown]
# ## Part 4 – Design Challenge (20 points)

# %%
import random
from collections import defaultdict, Counter

class FloatDict(defaultdict):
    def __init__(self):
        super().__init__(float)


class AveragedPerceptron:
    def __init__(self):
        # self.weights = defaultdict(lambda: defaultdict(float))
        self.weights = defaultdict(FloatDict)
        self.totals = defaultdict(float)
        self.timestamps = defaultdict(int)
        self.i = 0  # total updates

    def predict(self, features):
        scores = defaultdict(float)
        for feat, value in features.items():
            if value == 0 or feat not in self.weights:
                continue
            for label, weight in self.weights[feat].items():
                scores[label] += value * weight
        return max(scores, key=scores.get) if scores else 'O'

    def update(self, truth, guess, features):
        def upd_feat(feat, label, value):
            param = (feat, label)
            self.totals[param] += (self.i - self.timestamps[param]) * self.weights[feat][label]
            self.timestamps[param] = self.i
            self.weights[feat][label] += value

        self.i += 1
        if truth == guess:
            return
        for feat in features:
            upd_feat(feat, truth, 1.0)
            upd_feat(feat, guess, -1.0)

    def average_weights(self):
        for feat, weights in self.weights.items():
            for label in weights:
                param = (feat, label)
                total = self.totals[param] + (self.i - self.timestamps[param]) * weights[label]
                averaged = total / self.i
                if averaged:
                    self.weights[feat][label] = averaged
                else:
                    del self.weights[feat][label]


# %%
import string

def extract_features(sentence, index):
    word = sentence[index]
    features = {
        'bias': 1.0,
        'word.lower=' + word.lower(): 1.0,
        'word[-3:]=' + word[-3:]: 1.0,
        'word[-2:]=' + word[-2:]: 1.0,
        'word.isupper=' + str(word.isupper()): 1.0,
        'word.istitle=' + str(word.istitle()): 1.0,
        'word.isdigit=' + str(word.isdigit()): 1.0,
        'word.shape=' + word_shape(word): 1.0,
        'word.has_hyphen=' + str('-' in word): 1.0,
        'word.has_digit=' + str(any(char.isdigit() for char in word)): 1.0,
        'word.has_punct=' + str(any(char in string.punctuation for char in word)): 1.0,
    }

    if index > 0:
        prev = sentence[index - 1]
        features.update({
            '-1:word.lower=' + prev.lower(): 1.0,
            '-1:word.istitle=' + str(prev.istitle()): 1.0,
            '-1:word.shape=' + word_shape(prev): 1.0,
        })
    else:
        features['BOS'] = 1.0

    if index < len(sentence) - 1:
        next = sentence[index + 1]
        features.update({
            '+1:word.lower=' + next.lower(): 1.0,
            '+1:word.istitle=' + str(next.istitle()): 1.0,
            '+1:word.shape=' + word_shape(next): 1.0,
        })
    else:
        features['EOS'] = 1.0

    return features

def word_shape(word):
    shape = ''
    for char in word:
        if char.isupper():
            shape += 'X'
        elif char.islower():
            shape += 'x'
        elif char.isdigit():
            shape += 'd'
        else:
            shape += char
    return shape


# %%
def train_perceptron(dataset, epochs=5):
    model = AveragedPerceptron()
    for _ in range(epochs):
        random.shuffle(dataset)
        for sentence, tags in dataset:
            for i in range(len(sentence)):
                features = extract_features(sentence, i)
                pred = model.predict(features)
                model.update(tags[i], pred, features)
    model.average_weights()
    return model


# %%
def read_data(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        sent, labs = [], []
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    labels.append(labs)
                    sent, labs = [], []
                continue
            parts = line.split()
            if len(parts) == 2:
                token, label = parts
            else:
                token, label = parts[0], 'O'  # fallback if label is missing
            sent.append(token)
            labs.append(label)
        if sent:
            sentences.append(sent)
            labels.append(labs)
    return list(zip(sentences, labels))


# %%
def read_unlabeled_data(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sent = []
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    sent = []
                continue
            sent.append(line)
        if sent:
            sentences.append(sent)
    return sentences


# %%
train_file = "EN/train"
dev_in_file = "EN/dev.in"
gold_file = "EN/dev.out"

train_data = read_data(train_file)


# %%
model = train_perceptron(train_data, epochs=20)

# %%
import dill

def save_model(model, filename='part_4_model.pkl'):
    with open(filename, 'wb') as f:
        dill.dump(model, f)
    print(f"Model saved to {filename}")

# save_model(model)

# %%
def load_model(filename='model.pkl'):
    with open(filename, 'rb') as f:
        model = dill.load(f)
    print(f"Model loaded from {filename}")
    return model
# model = load_model("part_4_model.pkl")

# %% [markdown]
# Code to write and evaluate dev predictions

# %%
dev_sentences = read_unlabeled_data(dev_in_file)

# %%

with open("EN/dev.p4.out", 'w', encoding='utf-8') as out_f:
    for sent in dev_sentences:
        for i in range(len(sent)):
            feats = extract_features(sent, i)
            pred = model.predict(feats)
            out_f.write(f"{sent[i]} {pred}\n")
        out_f.write("\n")


# %%
import datetime

with open("eval_log.txt", "a") as f:
    f.write(f"\n----- Run at {datetime.datetime.now()} -----\n")

# !python EvalScript/evalResult.py EN/dev.out EN/dev.p4.out >> eval_log.txt


# %% [markdown]
# Code to write test predictions:

# %%
test_in_file = "EN/test.in"
test_sentences = read_unlabeled_data(test_in_file)

# %%
with open("EN/test.p4.out", 'w', encoding='utf-8') as out_f:
    for sent in test_sentences:
        for i in range(len(sent)):
            feats = extract_features(sent, i)
            pred = model.predict(feats)
            out_f.write(f"{sent[i]} {pred}\n")
        out_f.write("\n")



