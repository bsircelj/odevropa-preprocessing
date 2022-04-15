import os
import pandas as pd
import json
from langdetect import detect
import nltk.data
import numpy as np

sample = {
    "id": "50568",
    "title": "The Art of the Bone-Setter",
    "subtitle": "A Testimony and a Vindication",
    "author": "George Matthews Bennett",
    "release date": "November 28, 2015",
    "languages": ["en"],
    "text_en": "Produced by Turgut Dincer and The Online Distributed...",
    "annotations": ["Walking", "Shoulder", "Gentry", "Blacksmith", "Natural_rubber"],
    "date": 2015}


def test_check():
    test_dataset = pd.read_csv("../../data/test_gutenberg/13670.csv")
    print(test_dataset)


if __name__ == '__main__':
    test_check()
    jsonl_location = "../../data/gutenberg-combined"

    bad_id = []
    languages_present = []
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for root, dirs, files in os.walk(jsonl_location):
        for file_name in files:
            if ".jsonl" in file_name:
                print(f"Processing {file_name}")
                with open(f'{jsonl_location}/{file_name}', 'rb') as file:
                    sentences = np.array([])
                    book_id = np.array([])
                    for line_id, line in enumerate(file.readlines()):
                        #                         try:
                        file_data = json.loads(line)
                        text = file_data["text_en"]
                        detected_lang = detect(text)
                        if detected_lang != "en":
                            bad_id.append(file_data["id"])
                            languages_present.append(detected_lang)
                            continue
                        sentences = np.concatenate((sentences, sentence_tokenizer.tokenize(text)))
                        book_id = np.concatenate((book_id, np.full(np.shape(sentences), int(file_data["id"]))))
                        break
                        #                         except:
                        #                             print(f"Exception at line {line_id}")

                    dataset = pd.DataFrame({"sentence": sentences, "book_id": book_id})
                    print(dataset)
                    # dataset.to_csv(f"{location}/{file_name[:-6]}.csv")
