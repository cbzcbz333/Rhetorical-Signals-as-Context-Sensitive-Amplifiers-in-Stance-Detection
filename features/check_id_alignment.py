import json

def load_ids(path, key="id"):
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s.add(json.loads(line)[key])
    return s

rf_ids = load_ids("../data/rhetoric_features_all0113addcon2.jsonl")
for split in ["../data/train.jsonl", "../data/validation.jsonl", "../data/test.jsonl"]:
    ids = load_ids(split)
    missing = ids - rf_ids
    print(split, "n=", len(ids), "missing_in_rf=", len(missing))
    if missing:
        print("example missing:", list(missing)[:5])
