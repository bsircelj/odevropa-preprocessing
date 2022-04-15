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
from elastic_timeline_split import split_timeline

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

emotion_list = ['Happy', 'Surprised', 'Sad', 'Angry-Disgusted', 'Fearful', 'Other']
encode_dict = {label: i for i, label in enumerate(emotion_list)}

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


def emotion_processing(data, emotion_df, tale_name, split=10, confidence_threshold=0.8, draw_timeline=False):
    tale_data = emotion_df.loc[emotion_df["enTitle"] == tale_name]
    if len(tale_data.index) == 0:
        print(f"Tale {tale_name} missing")
        return False
    tale_data.loc[tale_data["probability"] < confidence_threshold, "relation"] = "Other"

    # filtered_emo = tale_data.loc[tale_data["probability"] > confidence_threshold]
    # data["emotions"] = list(f_emo["relation"])
    emo_array = np.array([])
    for row_index in tale_data.index:
        current_emo = encode_dict[tale_data.loc[row_index, "relation"]]
        ch_len = len(tale_data.loc[row_index, "sentence"])

        emo_array = np.concatenate((emo_array, np.full(ch_len, current_emo, np.uint8)))

    f_emo = pd.DataFrame({"relation": emo_array})

    length = len(f_emo.index)
    emotion_summary = dict()
    emotion_summary["total"] = len(tale_data[tale_data["relation"] != "Other"]) / len(tale_data.index)
    min_split = min(split, length)
    first = 0
    last = length

    emotion_time_line = []
    top_percentage = 0
    fig = go.Figure()

    temporary_summary = dict()
    for emotion_id, emotion in enumerate(emotion_list):
        if emotion == "Other":
            continue
        # Whole tale summary
        emo_percentage = len(f_emo[f_emo["relation"] == emotion_id]) / length
        emotion_summary[emotion] = emo_percentage
        if emo_percentage > top_percentage:
            top_percentage = emo_percentage
            data["top_emotion"] = emotion

        # Part based summary
        intensity_array = []
        indices = np.round(np.linspace(first, last, min_split + 1))
        for cur, _ in enumerate(indices[1:], 1):
            count = f_emo.loc[(f_emo["relation"] == emotion_id)
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
        if draw_timeline:
            fig.add_trace(go.Scatter(x=np.arange(len(interpolated_array)), y=interpolated_array, mode='lines',
                                     line_color=emotion_color[emotion_id], name=emotion))

    for e in temporary_summary.keys():
        timeline_summary[data["top_emotion"]][e] += temporary_summary[e]
    timeline_summary[data["top_emotion"]]["sum"] += 1

    if draw_timeline:
        fig.update_layout(title=tale_name)
        fig.write_html(f'../images/emotion_timeline_character/{tale_name}.html')

    data["emotion_summary"] = emotion_summary
    data["emotion_time_line"] = emotion_time_line

    return True


def olfactory_processing(data, olfactory_df, tale_name):
    tale_data = olfactory_df.loc[olfactory_df["enTitle"] == tale_name]
    data["olfactory_objects"] = list(tale_data["olfactory_object"])


def construct_elastic(emotions_location, olfactory_location, concepts_location, tales_locations,
                      output_data, output_timeline, draw_timeline=False):
    whole_data = []

    emotions_df = pd.read_csv(emotions_location, index_col="Unnamed: 0")
    olfactory_df = pd.read_csv(olfactory_location)
    # olfactory_df.drop_duplicates(subset='id', inplace=True)

    with open(concepts_location, 'r') as file:
        concepts = json.load(file)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for filename in tales_locations:
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

            valid = emotion_processing(data, emotions_df, data["name"], draw_timeline=draw_timeline)
            olfactory_processing(data, olfactory_df, data["name"])
            filtered_concepts = []
            for c in concepts[data["name"]]:
                if not any([s in c for s in blacklist]):
                    filtered_concepts.append(c)
            data["concepts"] = filtered_concepts

            if valid:
                whole_data.append(data)

    with open(output_data, 'w') as file:
        for data_instance in whole_data:
            file.write(f'{json.dumps(data_instance)}\n')

    split_timeline(output_data, output_timeline)

    # Draw timelines for each emotion
    if draw_timeline:
        for top_emotion in emotion_list:
            if top_emotion == "Other":
                continue
            fig = go.Figure()
            for emotion_id, emotion in enumerate(emotion_list):
                if emotion == "Other":
                    continue
                fig.add_trace(go.Scatter(x=np.arange(10), y=timeline_summary[top_emotion][emotion], mode='lines',
                                         line_color=emotion_color[emotion_id], name=emotion))

            fig.update_layout(title=top_emotion)
            fig.write_html(f'../images/emotion_timeline_summary/{top_emotion}.html')


if __name__ == '__main__':
    construct_elastic(emotions_location="../data/predictions_with_tile_id.csv",
                      olfactory_location="../data/olfactory_objects_in_smell_sentences_only.csv",

                      concepts_location="../data/concepts-per-tale.json",
                      tales_locations=["../data/tales-grimm.json", "../data/tales-andersen.json"],
                      # Processed tales data (turned into a json)
                      output_data=f"../data/elastic.jsonl",  # Output 1
                      output_timeline=f"../data/timeline.jsonl",  # Output 2
                      draw_timeline=False
                      )
