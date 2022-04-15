import numpy as np
import pandas as pd
from collections import Counter
import nltk.data

if __name__ == '__main__':
    # dataset = pd.read_csv("../../data/reman-emotions.csv", header=None)
    # column_names = ["index", "author", "name", "sentence", "emotion", ""]
    # at_dataset = pd.read_csv("../../data/utf-8-emotions-reman.txt.csv", header=None, sep="@")
    dataset = pd.read_csv("../../data/utf-8-emotions-reman.txt.csv", sep="@", on_bad_lines='skip')
    # print(at_dataset)
    ordered_labels = ["joy", "sadness", "anger", "fear", "surprise", "anticipation", "trust", "disgust"]
    encode_dict = {label: i for i, label in enumerate(ordered_labels)}

    dataset = dataset[dataset.relation.isin(ordered_labels)]

    dataset["ENCODE_CAT"] = [encode_dict[x] for x in dataset["relation"]]
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


    def get_middle(sentence):
        for i, sentence in enumerate(tokenizer.tokenize(sentence)):
            if i == 1:
                return sentence


    # dataset["sentence"] = [get_middle(x) for x in dataset["sentence"]]
    dataset["sentence"] = dataset["sentence"].apply(get_middle)
    split_point = int(len(dataset.index) * 0.8)
    train_dataset = dataset.loc[dataset.index[:split_point], :]
    test_dataset = dataset.loc[dataset.index[split_point:], :]
    # print(Counter(train_dataset.emotion))
    # print(Counter(test_dataset.emotion))

    train_dataset.to_csv("../../data/train_gutenberg.csv", header=True)
    test_dataset.to_csv("../../data/test_gutenberg.csv", header=True)
    # print(dataset)
