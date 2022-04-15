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
import urllib.parse
import plotly.graph_objects as go
import scipy

blacklist_old = ["Pakistan", "Internet", "Chicago", "Canada", "Cold War", "Jesus", "Adolf Hitler", "China", "Brexit",
                 "Latin", "Hebrew language", "Apollo 11", "COVID-19 pandemic", "World War", "Elizabeth II",
                 "Dissolution of the Soviet Union", "UNESCO", "Michael Jackson", "World War II", "Italian language",
                 "Weimar Republic", "Colombia", "Pakistan", "Cold War", "Film", "Soviet Union", "Obama",
                 "United States",
                 "Saddam Hussein", "Apollo", "Communism", "Charles II of England", "Iraq", "George W. Bush",
                 "Edward VIII",
                 "Forward (association football)", "Brexit", "Denmark", "Argentina", "Abraham Lincoln", "Washington",
                 "H. G. Wells", "Germany", "Poland", "Americas", "Qing", "George", "United States",
                 "Cuius regio, eius religio", "Indonesia", "Belgium", "Islam", "band", "Radiocarbon",
                 "French" "Victoria",
                 "Trump", "British Raj", "Gleichschaltung", "Broadway", "Rock music", "September 11 attacks",
                 "Copula (linguistics)", "England", "Russia", "Seven Years", "Heel (professional wrestling)",
                 "Hillary Clinton", "George VI", "Edward VII", "Queen (band)", "Henry IV", "Victorian" "France",
                 "Android", "Martin Luther King J", "bomb", "John F. Kennedy", "Farthing", "Italia", "United Kingdom",
                 "France", "Internet", "Latin", "Whatever"
                 ]

blacklist = ["Internet", "Chicago", "Cold War", "Jesus", "Adolf Hitler", "Brexit",
             "Latin", "Hebrew language", "Apollo 11", "COVID-19 pandemic", "World War", "Elizabeth II",
             "Dissolution of the Soviet Union", "UNESCO", "Michael Jackson", "World War II",
             "Weimar Republic", "Colombia", "Cold War", "Film", "Obama", "United States",
             "Saddam Hussein", "Apollo", "Communism", "Charles II of England", "George W. Bush", "Edward VIII",
             "Forward (association football)", "Brexit", "Abraham Lincoln", "Washington",
             "H. G. Wells", "Americas", "Qing", "George",
             "Cuius regio, eius religio", "Islam", "band", "Radiocarbon", "Victoria",
             "Trump", "British Raj", "Gleichschaltung", "Broadway", "Rock music", "September 11 attacks",
             "Copula (linguistics)", "Seven Years", "Heel (professional wrestling)",
             "Hillary Clinton", "George VI", "Edward VII", "Queen (band)", "Henry IV", "Victorian", "France",
             "Android", "Martin Luther King J", "bomb", "John F. Kennedy",
             "Internet", "Latin", "Whatever"
             ]

emotion_list = ['Happy', 'Surprised', 'Sad', 'Angry-Disgusted', 'Fearful']
emotion_color = ["green", "yellow", "blue", "red", "orange"]
timeline_summary = dict()
for top_emotion in emotion_list:
    timeline = dict()
    for emotion in emotion_list:
        timeline[emotion] = np.zeros(10)
    timeline["sum"] = 0
    timeline_summary[top_emotion] = timeline


def is_duplicate(data_array, name):
    for data_instance in data_array:
        if name.strip().lower() in data_instance["name"].strip().lower():
            print(f"Duplicate: {name}")
            return True

    return False


def emotion_processing(data, emotion_df, tale_name, split=10, confidence_threshold=0.8):
    tale_data = emotion_df.loc[emotion_df["enTitle"] == tale_name]
    if len(tale_data.index) == 0:
        print(f"Tale {tale_name} missing")
        return False

    f_emo = tale_data.loc[tale_data["probability"] > confidence_threshold]
    data["emotions"] = list(f_emo["relation"])

    filtered_length = len(f_emo.index)
    emotion_summary = dict()
    org_length = len(tale_data.index)
    min_split = min(split, org_length)
    first = tale_data.index[0]
    last = tale_data.index[-1] + 1

    emotion_time_line = []
    top_percentage = 0
    fig = go.Figure()

    temporary_summary = dict()
    for emotion_id, emotion in enumerate(emotion_list):

        # Whole tale summary
        emo_percentage = len(f_emo[f_emo["relation"] == emotion]) / filtered_length
        emotion_summary[emotion] = emo_percentage
        if emo_percentage > top_percentage:
            top_percentage = emo_percentage
            data["top_emotion"] = emotion

        # Part based summary
        intensity_array = []
        indices = np.round(np.linspace(first, last, min_split + 1))
        for cur, _ in enumerate(indices[1:], 1):
            count = f_emo.loc[(f_emo["relation"] == emotion)
                              & (f_emo.index >= indices[cur - 1])
                              & (f_emo.index < indices[cur])].shape[0]

            intensity_array.append(count / (indices[cur] - indices[cur - 1]))

        timestamps = np.linspace(0.1, 1, split)
        if not min_split == split:
            f1d = scipy.interpolate.interp1d(np.linspace(0.1, 1, min_split), intensity_array)
            interpolated_array = f1d(timestamps)
        else:
            interpolated_array = intensity_array

        for timestamp, intensity in zip(timestamps, interpolated_array):
            emotion_time_line.append({"timestamp": timestamp,
                                      "emotion": emotion,
                                      "intensity": intensity})

        temporary_summary[emotion] = np.copy(interpolated_array)
        # fig.add_trace(go.Scatter(x=np.arange(len(interpolated_array)), y=interpolated_array, mode='lines',
        #                          line_color=emotion_color[emotion_id], name=emotion))

    for e in temporary_summary.keys():
        timeline_summary[data["top_emotion"]][e] += temporary_summary[e]
    timeline_summary[data["top_emotion"]]["sum"] += 1

    # fig.update_layout(title=tale_name)
    # fig.write_html(f'./images/emotion_timeline/{tale_name}.html')

    emotion_summary["total"] = filtered_length / org_length
    data["emotion_summary"] = emotion_summary
    data["emotion_time_line"] = emotion_time_line

    return True


def olfactory_processing(data, olfactory_df, tale_name):
    tale_data = olfactory_df.loc[olfactory_df["enTitle"] == tale_name]
    data["olfactory_objects"] = list(tale_data["olfactory_object"])


def construct_elastic(emotions_location, olfactory_location):
    whole_data = []

    emotions_df = pd.read_csv(emotions_location, index_col="Unnamed: 0")
    olfactory_df = pd.read_csv(olfactory_location)
    olfactory_df.drop_duplicates(subset='id', inplace=True)

    with open("../data/archive/concepts-per-tale2.json", 'r') as file:
        concepts = json.load(file)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for filename in ["../data/archive/tales-grimm.json", "../data/archive/tales-andersen.json"]:
        with open(filename) as f:
            tales = json.load(f)

        for i, tale in enumerate(tales):
            print(i)
            data = dict()
            data["name"] = tale['enTitle']
            if is_duplicate(whole_data, data["name"]):
                continue

            data["text_en"] = ""
            data["sentences"] = []
            for paragraph in tale['comppars']:
                data["text_en"] = f'{data["text_en"]} {paragraph["enPar"]}'
                for sentence in tokenizer.tokenize(paragraph["enPar"]):
                    data["sentences"].append(sentence)

            valid = emotion_processing(data, emotions_df, data["name"])
            olfactory_processing(data, olfactory_df, data["name"])
            filtered_concepts = []
            for c in concepts[data["name"]]:
                if not any([s in c for s in blacklist]):
                    filtered_concepts.append(c)
            data["concepts"] = filtered_concepts

            if valid:
                whole_data.append(data)

    with open(f"../data/elastic.jsonl", 'w') as file:
        for data_instance in whole_data:
            file.write(f'{json.dumps(data_instance)}\n')

    # Draw timelines for each emotion

    # for top_emotion in emotion_list:
    #     fig = go.Figure()
    #     for emotion_id, emotion in enumerate(emotion_list):
    #         fig.add_trace(go.Scatter(x=np.arange(10), y=timeline_summary[top_emotion][emotion], mode='lines',
    #                                  line_color=emotion_color[emotion_id], name=emotion))
    #
    #     fig.update_layout(title=top_emotion)
    #     fig.write_html(f'./images/emotion_timeline_summary/{top_emotion}.html')


if __name__ == '__main__':
    # construct_elastic("../data/archive/predictions_with_tile_id.csv",
    #                   "../data/archive/predictions-olfactory-tale_olfactory_objects.csv")

    construct_elastic("../data/april_update/predictions_with_tile_id.csv",
                      "../data/april_update/olfactory_objects_in_smell_sentences_only.csv")
