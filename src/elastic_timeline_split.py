import json


def split_timeline(elastic_data_location, timeline_data_location):
    data = []
    with open(elastic_data_location, 'r') as file:
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

    with open(elastic_data_location, 'w') as file:
        for data_instance in data:
            file.write(f'{json.dumps(data_instance)}\n')

    with open(timeline_data_location, 'w') as file:
        for data_instance in timeline_data:
            file.write(f'{json.dumps(data_instance)}\n')


if __name__ == '__main__':
    split_timeline(elastic_data_location=f"../data/elastic.jsonl",
                   timeline_data_location=f"../data/timeline.jsonl")
