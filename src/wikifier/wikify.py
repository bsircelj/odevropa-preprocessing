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

def transform(filename):
    with open("../../data/archive/out.json", 'w') as write_file:
        write_file.write("[")
        with open(filename, 'r', encoding="ISO-8859-1") as f:
            for line in f.readlines():
                write_file.write(f'{line[:-2]}}},\n')
        write_file.write("]")


class Concept:
    def __init__(self, title, page_rank, cosine, url):
        self.title = title
        self.page_rank = page_rank
        self.cosine = cosine
        self.url = url

    def __lt__(self, other):
        return self.page_rank * self.cosine < other.page_rank * other.cosine

    def __str__(self):
        return f'{self.title[:48]:50} {self.page_rank:.6f} Cos: {self.cosine:.6f} {self.url}'

    def __eq__(self, other):
        return self.title == other.title


def load_concepts():
    df_load = pd.read_csv("../../data/archive/top_concepts.csv")
    con = np.array([])
    for _, row in df_load.iterrows():
        con = np.concatenate((con, [Concept(row['title'], row['page_rank'])]))
    return con


def wikify():
    concept_dict = dict()
    for filename in ["data/tales-grimm.json", "data/tales-andersen.json"]:
        with open(filename) as f:
            tales = json.load(f)

        # concepts = load_concepts()

        response = None
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        total_iterations = len(tales)
        iteration_no = 0
        start_time = time.time()

        # err_list = [6, 7, 9, 12, 14, 26, 42, 62, 71, 77, 78, 82, 84, 87, 94, 105, 106, 118, 124, 131, 135, 139, 151, 152]
        err_list2 = []
        # total_iterations = len(err_list)


        for i, tale in enumerate(tales):
            try:
                concepts = np.array([])

                print(f'tale {i}')
                run_time_seconds = int(time.time() - start_time)
                total_time = timedelta(
                    seconds=run_time_seconds * total_iterations // (iteration_no + 0.000001))
                time_left = total_time - timedelta(seconds=run_time_seconds)
                print("Progress {:4.1f}% time left: {}".format(iteration_no / total_iterations * 100,
                                                               time_left))
                iteration_no += 1

                word_soup = ""
                # sk = 0
                for paragraph in tale['comppars']:
                    word_soup = f'{word_soup} {paragraph["enPar"]}'
                    # sk += 1
                    # if sk > 10:
                    #     break
                split_tale = []
                if len(word_soup) > 10000:
                    current_part = ""
                    for sentence in tokenizer.tokenize(word_soup):
                        if len(current_part) + len(sentence) > 10000:
                            split_tale.append(current_part)
                            current_part = sentence
                        else:
                            current_part = f'{current_part} {sentence}'

                    split_tale.append(current_part)
                else:
                    split_tale.append(word_soup)

                for split_id, part in enumerate(split_tale):
                    request_body = {"userKey": "wcssnhtlcfhmoszgssoftrvaliewyj",
                                    "text": part,
                                    "lang": "en"}
                    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                    response = requests.post(url=f"http://www.wikifier.org/annotate-article", data=request_body,
                                             headers=headers)
                    # print(f'Status code: {response.status_code}')
                    content = json.loads(response.content)
                    # with open(f"./wiki_response/{tale['enTitle']}_part{split_id}.json", 'w') as file:
                    #     file.write(json.dumps(content, indent=4))
                    for entry in content['annotations']:
                        if entry['pageRank'] > 0.0005:
                            concepts = np.concatenate(
                                (concepts, [Concept(entry['title'], entry['pageRank'], entry['cosine'], entry['url'])]))

                concepts = np.unique(concepts)
                concepts = np.sort(concepts)
                concepts = np.flip(concepts)
                # df = pd.DataFrame(columns=['page_rank', 'title', 'url'])
                titles = []
                for i, c in enumerate(concepts):
                    # print(f'{i:3d} {c}')
                    # df.loc[i] = [c.page_rank, c.title, c.url]
                    titles.append(c.title)
                    if i >= 99:
                        break
                # df.to_csv('top_concepts3.csv')
                concept_dict[tale["enTitle"]] = titles
                # break

            except Exception:
                print(f'Error in tale {tale["enTitle"]}:')
                print(traceback.format_exc())
                err_list2.append(i)

    with open(f"../../data/concepts-per-tale2.json", 'w') as file:
        file.write(json.dumps(concept_dict, indent=4))


if __name__ == '__main__':
    # transform("tales-grimm-en-it.json")
    # sys.exit()
    wikify()
