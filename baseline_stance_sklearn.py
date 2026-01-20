# baseline_stance_sklearn.py

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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
# =========================
# 1. 读取数据
# =========================

DATA_PATH = "data/semeval2016_with_rhetorical_features1219addcon.csv"




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
CONTRAST_COLS = [
    "contrast_marker_count",
    "contrast_sentence_ratio",
    "has_contrast_structure"
]

NUMERIC_COLS = RQ_COLS + MODAL_COLS + CONTRAST_COLS

#NUMERIC_COLS = RQ_COLS + MODAL_COLS

# =========================
# 读取数据 II
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# 数值特征清洗（关键一步）
# =========================
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric_column)
# =========================
# 切分数据
# =========================
# 只用 train / test（SemEval 官方划分）
train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

X_train = train_df
y_train = train_df["stance"]

X_test = test_df
y_test = test_df["stance"]

print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")


# =========================
# 3. 公共文本特征（TF-IDF）
# =========================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    stop_words="english"
)

# =========================
# 4. Baseline 0: Text only
# =========================

print("\n===== Baseline B0: Text only =====")

pipe_b0 = Pipeline([
    ("tfidf", tfidf),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

pipe_b0.fit(X_train[TEXT_COL], y_train)
pred_b0 = pipe_b0.predict(X_test[TEXT_COL])

print(classification_report(y_test, pred_b0, digits=4))
print("Macro-F1:", f1_score(y_test, pred_b0, average="macro"))

# =========================
# 5. Baseline 1: Text + Rhetorical Question
# =========================

print("\n===== Baseline B1: Text + Rhetorical Question =====")

preprocess_b1 = ColumnTransformer(
    transformers=[
        ("text", tfidf, TEXT_COL),
        ("rq", Pipeline([
            ("scaler", StandardScaler())
        ]), RQ_COLS)
    ]
)

pipe_b1 = Pipeline([
    ("features", preprocess_b1),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

pipe_b1.fit(X_train, y_train)
pred_b1 = pipe_b1.predict(X_test)

print(classification_report(y_test, pred_b1, digits=4))
print("Macro-F1:", f1_score(y_test, pred_b1, average="macro"))

# =========================
# 6. Baseline 2: Text + RQ + Modality
# =========================

print("\n===== Baseline B2: Text + RQ + Modality =====")

preprocess_b2 = ColumnTransformer(
    transformers=[
        ("text", tfidf, TEXT_COL),
        ("rq", Pipeline([
            ("scaler", StandardScaler())
        ]), RQ_COLS),
        ("modal", Pipeline([
            ("scaler", StandardScaler())
        ]), MODAL_COLS)
    ]
)

pipe_b2 = Pipeline([
    ("features", preprocess_b2),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

pipe_b2.fit(X_train, y_train)
pred_b2 = pipe_b2.predict(X_test)

print(classification_report(y_test, pred_b2, digits=4))
print("Macro-F1:", f1_score(y_test, pred_b2, average="macro"))

print("\n===== Baseline B3: Text + RQ + Modality + Contrast =====")

preprocess_b3 = ColumnTransformer(
    transformers=[
        ("text", tfidf, TEXT_COL),

        ("rq", Pipeline([
            ("scaler", StandardScaler())
        ]), RQ_COLS),

        ("modal", Pipeline([
            ("scaler", StandardScaler())
        ]), MODAL_COLS),

        ("contrast", Pipeline([
            ("scaler", StandardScaler())
        ]), CONTRAST_COLS)
    ]
)

pipe_b3 = Pipeline([
    ("features", preprocess_b3),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

pipe_b3.fit(X_train, y_train)
pred_b3 = pipe_b3.predict(X_test)

print(classification_report(y_test, pred_b3, digits=4))
print("Macro-F1:", f1_score(y_test, pred_b3, average="macro"))
