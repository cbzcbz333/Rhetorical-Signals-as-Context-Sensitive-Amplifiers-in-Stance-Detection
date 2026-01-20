import numpy as np, json

zeros = 0
total = 0
with open("../data/rhetoric_features_all0113addcon2.jsonl","r",encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        rf = np.array(obj["rf"])
        total += 1
        if np.all(rf == 0):
            zeros += 1

print("all-zero ratio:", zeros/total)
