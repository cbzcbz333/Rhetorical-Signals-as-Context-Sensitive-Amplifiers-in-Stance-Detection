import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# =========================
# 1. 读取数据
# =========================

DATA_PATH = "data/semeval2016_with_rhetorical_features1219add1.csv"
df = pd.read_csv(DATA_PATH)

# =========================
# 2. 特征列定义
# =========================

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

NUMERIC_COLS = RQ_COLS + MODAL_COLS

# =========================
# 3. 数值清洗（作用于 df）
# =========================

def clean_numeric_column(x):
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
        df[col] = df[col].apply(clean_numeric_column)

# =========================
# 4. 模型组件（统一）
# =========================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000,
    min_df=2,
    stop_words="english"
)

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

# =========================
# 5. 按 topic 跑 baseline
# =========================

results = []

for topic in sorted(df["topic"].unique()):
    print(f"\n===== Topic: {topic} =====")

    topic_df = df[df["topic"] == topic]

    train_df = topic_df[topic_df["split"] == "train"]
    test_df  = topic_df[topic_df["split"] == "test"]

    if len(train_df) < 100 or len(test_df) < 50:
        print("样本过少，跳过")
        continue

    X_train = train_df
    y_train = train_df["stance"]
    X_test  = test_df
    y_test  = test_df["stance"]

    # -------- B0 --------
    pipe_b0 = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])

    pipe_b0.fit(X_train[TEXT_COL], y_train)
    pred_b0 = pipe_b0.predict(X_test[TEXT_COL])
    f1_b0 = f1_score(y_test, pred_b0, average="macro")

    # -------- B1 --------
    preprocess_b1 = ColumnTransformer([
        ("text", tfidf, TEXT_COL),
        ("rq", Pipeline([
            ("scaler", StandardScaler())
        ]), RQ_COLS)
    ])

    pipe_b1 = Pipeline([
        ("features", preprocess_b1),
        ("clf", clf)
    ])

    pipe_b1.fit(X_train, y_train)
    pred_b1 = pipe_b1.predict(X_test)
    f1_b1 = f1_score(y_test, pred_b1, average="macro")

    # -------- B2 --------
    preprocess_b2 = ColumnTransformer([
        ("text", tfidf, TEXT_COL),
        ("rq", Pipeline([
            ("scaler", StandardScaler())
        ]), RQ_COLS),
        ("modal", Pipeline([
            ("scaler", StandardScaler())
        ]), MODAL_COLS)
    ])

    pipe_b2 = Pipeline([
        ("features", preprocess_b2),
        ("clf", clf)
    ])

    pipe_b2.fit(X_train, y_train)
    pred_b2 = pipe_b2.predict(X_test)
    f1_b2 = f1_score(y_test, pred_b2, average="macro")

    print(f"B0: {f1_b0:.4f} | B1: {f1_b1:.4f} | B2: {f1_b2:.4f}")

    results.append({
        "topic": topic,
        "B0_text": f1_b0,
        "B1_text+RQ": f1_b1,
        "B2_text+RQ+Mod": f1_b2
    })

# =========================
# 6. 汇总保存
# =========================

result_df = pd.DataFrame(results)
print("\n===== Summary =====")
print(result_df)

result_df.to_csv("results/baseline_macroF1_by_topic.csv", index=False)
print("\n✅ 已保存：results/baseline_macroF1_by_topic.csv")
