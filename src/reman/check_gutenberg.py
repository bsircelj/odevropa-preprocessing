import os
import pandas as pd
import numpy as np

folder_root = 'gutenberg-output'
emotions = ["joy", "sadness", "anger", "fear", "surprise", "anticipation", "trust", "disgust"]
ordered_labels = ["joy", "sadness", "anger", "fear", "surprise", "anticipation", "trust", "disgust"]


def check():
    for subfolder in [12, 13, 14]:
        for dirpath, dnames, fnames in os.walk(f'{folder_root}/{subfolder}'):
            for f in fnames:
                data = pd.read_csv(f'{dirpath}/{f}')
                for emotion in emotions[::-1]:
                    for sentence in data.loc[data[emotion] > 0.80, 'sentences']:
                        print(f'{emotion}:\n {sentence}\n____________________\n')


def print_example(text, targets, predicted):
    stats = '|'.join([f'{e} {t:d}:{p:.2f}' for e, t, p in zip(ordered_labels, targets, predicted)])
    print(f'{text}\n{stats}\n_____________')


def valid_print():
    text = "It is already a splendid monument of British benevolence; but is only a portion of the original plan, which is to complete another front towards Hyde Park; this will extend even further than the old hospital."
    predicted = [0.5, 0.2, 0.3, 0.8, 0.1, 0.1, 0.2, 0.9]
    targets = [1, 0, 0, 0, 0, 1, 0, 0]
    print_example(text, targets, predicted)
    print_example(text, targets, predicted)
    print_example(text, targets, predicted)
    print_example(text, targets, predicted)


if __name__ == '__main__':
    # check()
    valid_print()
