import pandas as pd
import json

# 读你的表（按实际路径改）
df = pd.read_csv("../data/semeval2016_with_rhetorical_features1219addcon.csv")

# 如果没有 id，就补一个
if "id" not in df.columns:
    df["id"] = df.index.astype(str)

rf_cols = [
    "question_count",
    "rhetorical_question_count",
    "rhetorical_question_ratio",
    "modal_verb_count",
    "hedge_marker_count",
    "strong_assertion_count",
    "epistemic_strength_score",
    "contrast_marker_count",
    "contrast_sentence_ratio",
    "has_contrast_structure"
]
count_cols = [
    "question_count",
    "rhetorical_question_count",
    "modal_verb_count",
    "hedge_marker_count",
    "strong_assertion_count",
    "contrast_marker_count"
]
# 缺失值兜底
df[rf_cols] = df[rf_cols].fillna(0)
# 1) 统一去掉字符串里的空格（如果有）
for c in rf_cols:
    df[c] = df[c].astype(str).str.strip()

# 2) 强制转数值：无法转换的变成 NaN
for c in rf_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 3) NaN 填 0
df[rf_cols] = df[rf_cols].fillna(0)

# 4) clip（这时一定是数值了）
#df[count_cols] = df[count_cols].clip(lower=0, upper=5)


# （可选）对计数做截断，减少极端值影响
count_cols = ["question_count","rhetorical_question_count","modal_verb_count","hedge_marker_count","strong_assertion_count","contrast_marker_count"]
print(df[count_cols].dtypes)
df[count_cols] = df[count_cols].clip(lower=0, upper=5)

out_path = "../data/rhetoric_features_all0113addcon2.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        #rf = [float(row[c]) for c in rf_cols]
        rf = [
            1.0 if row["question_count"] > 0 else 0.0,
            1.0 if row["rhetorical_question_count"] > 0 else 0.0,
            float(row["rhetorical_question_ratio"]),
            1.0 if row["modal_verb_count"] > 0 else 0.0,
            1.0 if row["hedge_marker_count"] > 0 else 0.0,
            1.0 if row["strong_assertion_count"] > 0 else 0.0,
            float(row["epistemic_strength_score"]),
            1.0 if row["contrast_marker_count"] > 0 else 0.0,
            float(row["contrast_sentence_ratio"]),
            float(row["has_contrast_structure"])
        ]
        f.write(json.dumps({"id": str(row["id"]), "rf": rf}, ensure_ascii=False) + "\n")

print("saved:", out_path, "rf_dim=", len(rf_cols))
