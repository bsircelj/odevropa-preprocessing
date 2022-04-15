import json
import sys
import requests
from collections import Counter
import numpy as np
import traceback
import pandas as pd
import time
from datetime import datetime, timedelta
import nltk.data

# #1. wikify the English tales from the given file (in attachement)
#
# here use the default configuration, then extract the titles of top 100
# concepts, and assign it to "concepts" attribute (to use it later in
# dashboard). keep the rest of wikification info as well.

if __name__ == '__main__':
    # transform()
    # sys.exit()
    # df_prior = pd.read_csv("data/sentences.csv", index_col='id')
    df = pd.DataFrame(columns=["sentence", "enTitle"])
    for location in ["data/tales-andersen.json", "data/tales-grimm.json"]:
        with open(location) as f:
            tales = json.load(f)

        total_iterations = len(tales)
        iteration_no = 0
        start_time = time.time()

        sentence_db = []
        titles = []

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        for i, tale in enumerate(tales):
            print(f'tale {i}')
            run_time_seconds = int(time.time() - start_time)
            total_time = timedelta(
                seconds=run_time_seconds * total_iterations // (iteration_no + 0.000001))
            time_left = total_time - timedelta(seconds=run_time_seconds)
            print("Progress {:4.1f}% time left: {}".format(iteration_no / total_iterations * 100,
                                                           time_left))
            iteration_no += 1

            for paragraph in tale['comppars']:
                for sentence in tokenizer.tokenize(paragraph["enPar"]):
                    sentence_db.append(sentence)
                    titles.append(tale['enTitle'])

        df_new = pd.DataFrame({'sentence': sentence_db, 'enTitle': titles})
        # df_new.index.name = 'id'
        df = pd.concat([df, df_new], ignore_index=True, sort=False)
    # df['id'] = df.index
    df.to_csv('sentences3.csv')
