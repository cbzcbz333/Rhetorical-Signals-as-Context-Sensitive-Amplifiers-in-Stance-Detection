import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline_piplines import pipe_b3
from baseline_stance_sklearn import clean_numeric_column
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

pipe_b3.fit(X_train, y_train)

coef = pipe_b3.named_steps["clf"].coef_
feature_names = pipe_b3.named_steps["features"].get_feature_names_out()

weights = pd.DataFrame(coef, columns=feature_names)


rhetorical_cols = [
    c for c in feature_names
    if c.startswith(("rq__", "modal__", "contrast__"))
]

#print(feature_names)

weights_rhetoric = weights[rhetorical_cols]

mean_abs_weight = weights_rhetoric.abs().mean(axis=0)
top_features = mean_abs_weight.sort_values(ascending=False)

top_features.plot(kind="barh", figsize=(6,4))
# Note: weights are extracted from a re-trained model with identical
# configuration to the experimental setting, for interpretability only.
ax = top_features.plot(kind="barh", figsize=(6,4))
ax.set_xlabel("Mean Absolute Weight")
ax.set_title("Rhetorical Feature Importance (B3)")

plt.tight_layout()
plt.savefig("results/figure2_rhetorical_feature_weights600.pdf")
plt.savefig("results/figure2_rhetorical_feature_weights600.png", dpi=600)
plt.show()
"""
C:\Users\27981\PycharmProjects\lichang\.venv\Scripts\python.exe C:\Users\27981\PycharmProjects\lichang\barchart.py 
Train size: 2620 | Test size: 1249

===== Baseline B0: Text only =====
              precision    recall  f1-score   support

     AGAINST     0.7976    0.5622    0.6596       715
       FAVOR     0.4609    0.5625    0.5067       304
        NONE     0.4091    0.6652    0.5066       230

    accuracy                         0.5813      1249
   macro avg     0.5559    0.5967    0.5576      1249
weighted avg     0.6441    0.5813    0.5942      1249

Macro-F1: 0.5576153990562719

===== Baseline B1: Text + Rhetorical Question =====
              precision    recall  f1-score   support

     AGAINST     0.7920    0.5538    0.6519       715
       FAVOR     0.4565    0.5526    0.5000       304
        NONE     0.4016    0.6652    0.5008       230

    accuracy                         0.5741      1249
   macro avg     0.5500    0.5906    0.5509      1249
weighted avg     0.6385    0.5741    0.5871      1249

Macro-F1: 0.5508900608191388

===== Baseline B2: Text + RQ + Modality =====
              precision    recall  f1-score   support

     AGAINST     0.7932    0.5524    0.6513       715
       FAVOR     0.4581    0.5757    0.5102       304
        NONE     0.3902    0.6261    0.4808       230

    accuracy                         0.5717      1249
   macro avg     0.5472    0.5847    0.5474      1249
weighted avg     0.6374    0.5717    0.5855      1249

Macro-F1: 0.5474277469232749

===== Baseline B3: Text + RQ + Modality + Contrast =====
              precision    recall  f1-score   support

     AGAINST     0.7920    0.5538    0.6519       715
       FAVOR     0.4577    0.5691    0.5073       304
        NONE     0.3827    0.6174    0.4725       230

    accuracy                         0.5693      1249
   macro avg     0.5441    0.5801    0.5439      1249
weighted avg     0.6353    0.5693    0.5837      1249

Macro-F1: 0.5439096624075065
"""