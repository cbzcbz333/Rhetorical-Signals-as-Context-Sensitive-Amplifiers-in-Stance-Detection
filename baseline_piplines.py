# baseline_pipelines.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# =========================
# 公共列定义
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

# =========================
# 公共 TF-IDF
# =========================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    stop_words="english"
)

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

tfidf_topic = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=1,               # 👈 关键
    stop_words="english"
)

# =========================
# B0: Text only
# =========================

pipe_b0 = Pipeline([
    ("tfidf", tfidf),
    ("clf", clf)
])

# =========================
# B1: Text + Rhetorical Question
# =========================

pipe_b1 = Pipeline([
    ("features", ColumnTransformer([
        ("text", tfidf, TEXT_COL),
        ("rq", Pipeline([
            ("scaler", StandardScaler())
        ]), RQ_COLS)
    ])),
    ("clf", clf)
])

# =========================
# B2: Text + RQ + Modality
# =========================

pipe_b2 = Pipeline([
    ("features", ColumnTransformer([
        ("text", tfidf, TEXT_COL),
        ("rq", Pipeline([
            ("scaler", StandardScaler())
        ]), RQ_COLS),
        ("modal", Pipeline([
            ("scaler", StandardScaler())
        ]), MODAL_COLS)
    ])),
    ("clf", clf)
])

# =========================
# B3: Text + RQ + Modality + Contrast
# =========================

pipe_b3 = Pipeline([
    ("features", ColumnTransformer([
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
    ])),
    ("clf", clf)
])
