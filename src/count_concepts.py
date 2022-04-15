from collections import Counter
import json

if __name__ == "__main__":
    with open("../data/archive/concepts-per-tale2.json", 'r') as file:
        concepts = json.load(file)

    cnt = Counter()

    for tale in concepts.keys():
        for c in concepts[tale]:
            cnt[c] += 1

    for c in cnt.most_common():
        print(c)
