import numpy as np
import pandas as pd
from collections import Counter
import nltk.data
import pickle

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

    current_sentence = dataset.loc[dataset.index[0], "sentence"]
    current_targets = np.zeros(len(ordered_labels))
    sentences = np.array([])
    targets = None

    for i in dataset.index:
        new_sentence = dataset.loc[i, "sentence"]
        if new_sentence != current_sentence:
            sentences = np.concatenate((sentences, [current_sentence]))
            if targets is None:
                targets = np.array([current_targets])
            else:
                targets = np.concatenate((targets, [current_targets]))
            current_sentence = new_sentence
            current_targets = np.zeros(len(ordered_labels))

        current_targets[encode_dict[dataset.loc[i, "relation"]]] = 1

    with open("sentences.pickle", "wb") as file:
        pickle.dump(sentences, file)

    with open("targets.pickle", "wb") as file:
        pickle.dump(targets, file)
