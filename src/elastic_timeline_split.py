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


def split_timeline():
    data = []
    with open(f"../data/elastic.jsonl", 'r') as file:
        for line in file.readlines():
            data.append(json.loads(line))

    timeline_data = []
    for instance in data:
        for i in instance["emotion_time_line"]:
            timeline_data.append({
                "tale": instance["name"],
                "top_tale_emotion": instance["top_emotion"],
                "emotion": i["emotion"],
                "intensity": i["intensity"],
                "timestamp": i["timestamp"],
                "olfactory_objects": instance["olfactory_objects"],
                "concepts": instance["concepts"],
                "text_en": instance["text_en"]
            })
        del instance["emotion_time_line"]

    with open(f"../data/elastic.jsonl", 'w') as file:
        for data_instance in data:
            file.write(f'{json.dumps(data_instance)}\n')

    with open(f"../data/timeline.jsonl", 'w') as file:
        for data_instance in timeline_data:
            file.write(f'{json.dumps(data_instance)}\n')


if __name__ == '__main__':
    split_timeline()
