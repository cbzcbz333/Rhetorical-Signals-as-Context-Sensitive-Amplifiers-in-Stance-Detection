import pandas as pd
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from baseline_piplines import pipe_b0, pipe_b1, pipe_b2, pipe_b3
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
tfidf_topic = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=1,              # 👈 关键：topic 内
    stop_words="english"
)

TEXT_COL = "text"

RQ_COLS = [
    "question_count",
    "rhetorical_question_count",
    "rhetorical_question_ratio"
]

MODAL_COLS = [
    "modal_verb_count",
    "hedge_marker_count",
    "strong_assertion_count",
    "epistemic_strength_score"
]

CONTRAST_COLS = [
    "contrast_marker_count",
    "contrast_sentence_ratio",
    "has_contrast_structure"
]

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

# -------- B0 --------
pipe_b0 = Pipeline([
    ("tfidf", tfidf_topic),
    ("clf", clf)
])

# -------- B1 --------
pipe_b1 = Pipeline([
    ("features", ColumnTransformer([
        ("text", tfidf_topic, TEXT_COL),
        ("rq", StandardScaler(), RQ_COLS)
    ])),
    ("clf", clf)
])

# -------- B2 --------
pipe_b2 = Pipeline([
    ("features", ColumnTransformer([
        ("text", tfidf_topic, TEXT_COL),
        ("rq", StandardScaler(), RQ_COLS),
        ("modal", StandardScaler(), MODAL_COLS)
    ])),
    ("clf", clf)
])

# -------- B3 --------
pipe_b3 = Pipeline([
    ("features", ColumnTransformer([
        ("text", tfidf_topic, TEXT_COL),
        ("rq", StandardScaler(), RQ_COLS),
        ("modal", StandardScaler(), MODAL_COLS),
        ("contrast", StandardScaler(), CONTRAST_COLS)
    ])),
    ("clf", clf)
])

# ===== 0. 读取数据 =====
DATA_PATH = "data/semeval2016_with_rhetorical_features1219addcon.csv"
df = pd.read_csv(DATA_PATH)
# =========================
# 数值特征清洗（topic-level 安全版）
# =========================

NUMERIC_COLS = (
    RQ_COLS +
    MODAL_COLS +
    CONTRAST_COLS
)

def clean_numeric(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, str):
        if x.lower() == "false":
            return 0
        if x.lower() == "true":
            return 1
        try:
            return float(x)
        except:
            return 0
    if pd.isna(x):
        return 0
    return x

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

topics = sorted(df["topic"].unique())

# ===== 1. 定义一个评估函数（复用你已有 pipeline）=====
def eval_pipeline(pipe, train_df, test_df):
    y_train = train_df["stance"]
    y_test = test_df["stance"]

    # 🔑 关键修复点
    if "tfidf" in pipe.named_steps:
        # B0：只喂 text 列
        X_train = train_df["text"]
        X_test = test_df["text"]
    else:
        # B1–B3：喂整个 DataFrame（ColumnTransformer 会自己取列）
        X_train = train_df
        X_test = test_df

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return f1_score(y_test, pred, average="macro")


# ===== 2. 存结果 =====
records = []

for topic in topics:
    print(f"\n===== Topic: {topic} =====")

    sub = df[df["topic"] == topic]
    train_df = sub[sub["split"] == "train"]
    test_df = sub[sub["split"] == "test"]

    if len(test_df) == 0:
        print("⚠️ No test data, skip")
        continue

    f1_b0 = eval_pipeline(pipe_b0, train_df, test_df)
    f1_b1 = eval_pipeline(pipe_b1, train_df, test_df)
    f1_b2 = eval_pipeline(pipe_b2, train_df, test_df)
    f1_b3 = eval_pipeline(pipe_b3, train_df, test_df)

    print(f"B0: {f1_b0:.4f} | B1: {f1_b1:.4f} | B2: {f1_b2:.4f} | B3: {f1_b3:.4f}")

    records.append({
        "topic": topic,
        "B0_text": f1_b0,
        "B1_text+RQ": f1_b1,
        "B2_text+RQ+Mod": f1_b2,
        "B3_text+RQ+Mod+Con": f1_b3
    })

# ===== 3. 汇总保存 =====
res_df = pd.DataFrame(records)
print("\n===== Summary (Macro-F1 by topic) =====")
print(res_df)

OUT_PATH = "results/baseline_macroF1_by_topic_B0_B3.csv"
res_df.to_csv(OUT_PATH, index=False)
print(f"\n✅ 已保存：{OUT_PATH}")

"""
C:\Users\27981\PycharmProjects\lichang\.venv\Scripts\python.exe C:\Users\27981\PycharmProjects\lichang\baseline_by_topic_full.py 

===== Topic: abortion =====
B0: 0.5815 | B1: 0.5810 | B2: 0.5521 | B3: 0.5383

===== Topic: atheism =====
B0: 0.4897 | B1: 0.4971 | B2: 0.4496 | B3: 0.4651

===== Topic: climate =====
B0: 0.4542 | B1: 0.4409 | B2: 0.4387 | B3: 0.4147

===== Topic: feminist =====
B0: 0.4974 | B1: 0.4975 | B2: 0.4868 | B3: 0.4897

===== Topic: hillary =====
B0: 0.5441 | B1: 0.5494 | B2: 0.5479 | B3: 0.5547

===== Summary (Macro-F1 by topic) =====
      topic   B0_text  B1_text+RQ  B2_text+RQ+Mod  B3_text+RQ+Mod+Con
0  abortion  0.581530    0.580987        0.552129            0.538295
1   atheism  0.489724    0.497050        0.449623            0.465082
2   climate  0.454212    0.440908        0.438738            0.414678
3  feminist  0.497396    0.497489        0.486766            0.489692
4   hillary  0.544096    0.549403        0.547890            0.554670
"""