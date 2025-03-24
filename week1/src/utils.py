import pandas as pd
from sklearn.utils import shuffle


def load_dataset(file_path):
    data=pd.read_csv(file_path)
    return shuffle(data,random_state=42)

def unique_values(data,column):
    return data[column].unique()

def split_dataset(data, column, value):
    true_branch = data[data[column] == value]
    false_branch = data[data[column] != value]
    return true_branch, false_branch

def most_common_label(labels):
    return labels.value_counts().idxmax()


def entropy(labels):
    from math import log2
    from collections import Counter
    entropy=0
    label_counts=Counter(labels)
    for label in label_counts:
        p=label_counts[label]/len(labels)
        entropy-=p*log2(p)
    return entropy

def best_split(data, target):
    best_gain = 0
    best_column = None
    best_value = None


    current_entropy = entropy(data[target])

    for col in data.columns:
        if col == target:
            continue

        values = unique_values(data, col)
        for value in values:
            left_split = data[data[col] == value]
            right_split = data[data[col] != value]

            if len(left_split) == 0 or len(right_split) == 0:
                continue

            p_left = len(left_split) / len(data)
            p_right = 1 - p_left
            gain = current_entropy - (p_left * entropy(left_split[target]) + p_right * entropy(right_split[target]))

            if gain > best_gain:
                best_gain, best_column, best_value = gain, col, value

    return best_column, best_value

