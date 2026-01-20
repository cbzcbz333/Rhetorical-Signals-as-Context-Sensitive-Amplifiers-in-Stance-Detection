import json
from features.rhetoric import RhetoricFeatureEncoder

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main(in_path: str, out_path: str):
    enc = RhetoricFeatureEncoder(cache_enabled=True)

    with open(out_path, "w", encoding="utf-8") as w:
        for row in load_jsonl(in_path):
            rid = row["id"]
            text = row["text"]
            rf = enc.extract_vector(text)
            w.write(json.dumps({"id": rid, "rf": rf}, ensure_ascii=False) + "\n")

    print("rf_dim =", enc.rf_dim, "saved to", out_path)

if __name__ == "__main__":
    main("train.jsonl", "rhetoric_features_train.jsonl")
