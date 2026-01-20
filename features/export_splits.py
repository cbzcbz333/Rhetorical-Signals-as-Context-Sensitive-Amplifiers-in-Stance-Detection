import json
import pandas as pd

# ====== 1) 读入你的表 ======
# 如果是 CSV
df = pd.read_csv("../data/semeval2016_with_rhetorical_features1219addcon.csv")

# 如果是 Excel，就用这行替换上面：
# df = pd.read_excel("processed_with_features.xlsx", engine="openpyxl")

# ====== 2) 确保有 id（用原始行号，千万别 reset_index） ======
if "id" not in df.columns:
    df["id"] = df.index.astype(str)
else:
    df["id"] = df["id"].astype(str)

# ====== 3) stance -> label 映射（SemEval 常用 3 类） ======
label_map = {"AGAINST": 0, "FAVOR": 1, "NONE": 2}
df["label"] = df["stance"].map(label_map)

# 如果你的 stance 里有其它值（比如 UNK），这里会出现 NaN
bad = df[df["label"].isna()][["id", "stance"]].head(10)
if len(bad) > 0:
    raise ValueError(f"存在无法映射的 stance 值（示例前10条）：\n{bad}")

df["label"] = df["label"].astype(int)

# ====== 4) 导出函数 ======
def dump_split(split_name: str, out_path: str):
    sub = df[df["split"] == split_name].copy()
    if len(sub) == 0:
        raise ValueError(f"split={split_name} 没有数据，请检查 split 列取值。")

    with open(out_path, "w", encoding="utf-8") as w:
        for _, r in sub.iterrows():
            w.write(json.dumps({
                "id": r["id"],
                "text": r["text"],
                "label": int(r["label"]),
                "topic": r["topic"],
                "stance": r["stance"],
                "split": r["split"]
            }, ensure_ascii=False) + "\n")

    print(f"saved {out_path} | n={len(sub)} | id_range=[{sub['id'].iloc[0]}..{sub['id'].iloc[-1]}]")

# ====== 5) 导出 train/validation/test ======
dump_split("train", "../data/train.jsonl")
dump_split("validation", "../data/validation.jsonl")
dump_split("test", "../data/test.jsonl")
