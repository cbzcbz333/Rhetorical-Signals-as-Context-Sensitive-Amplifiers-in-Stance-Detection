import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# =============== 1) 路径 ===============
DATA_PATH = "data/semeval2016_with_rhetorical_features1219addcon.csv"  # 改成你的最新文件

# =============== 2) 列配置 ===============
TEXT_COL = "text"
LABEL_COL = "stance"
TOPIC_COL = "topic"

RQ_COLS = ["question_count", "rhetorical_question_count", "rhetorical_question_ratio"]
MODAL_COLS = ["modal_verb_count", "hedge_marker_count", "strong_assertion_count", "epistemic_strength_score"]
CONTRAST_COLS = ["contrast_marker_count", "contrast_sentence_ratio", "has_contrast_structure"]

NUMERIC_COLS = RQ_COLS + MODAL_COLS + CONTRAST_COLS

def clean_numeric_column(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, str):
        xl = x.lower()
        if xl == "false": return 0
        if xl == "true":  return 1
        try:
            return float(x)
        except:
            return 0
    if pd.isna(x):
        return 0
    return x

# =============== 3) 读数据 + 清洗数值 ===============
df = pd.read_csv(DATA_PATH)
for c in NUMERIC_COLS:
    if c in df.columns:
        df[c] = df[c].apply(clean_numeric_column)

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

X_train, y_train = train_df, train_df[LABEL_COL]
X_test,  y_test  = test_df,  test_df[LABEL_COL]

# =============== 4) 定义 B0 / B3 pipeline（与你 baseline 保持一致） ===============
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    stop_words="english"
)

clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)

pipe_b0 = Pipeline([
    ("tfidf", tfidf),
    ("clf", clf)
])

preprocess_b3 = ColumnTransformer(
    transformers=[
        ("text", tfidf, TEXT_COL),
        ("rq", Pipeline([("scaler", StandardScaler())]), RQ_COLS),
        ("modal", Pipeline([("scaler", StandardScaler())]), MODAL_COLS),
        ("con", Pipeline([("scaler", StandardScaler())]), CONTRAST_COLS),
    ]
)

pipe_b3 = Pipeline([
    ("features", preprocess_b3),
    ("clf", clf)
])

# =============== 5) 训练 + 预测 ===============
pipe_b0.fit(X_train[TEXT_COL], y_train)
pred_b0 = pipe_b0.predict(X_test[TEXT_COL])

pipe_b3.fit(X_train, y_train)
pred_b3 = pipe_b3.predict(X_test)

# =============== 6) 找 B0对B3错 的样本 ===============
err_mask = (pred_b0 == y_test.values) & (pred_b3 != y_test.values)
err_df = X_test.loc[err_mask].copy()
err_df["gold"] = y_test.loc[err_mask].values
err_df["pred_b0"] = pred_b0[err_mask]
err_df["pred_b3"] = pred_b3[err_mask]

print(f"Found {len(err_df)} cases where B0 correct but B3 wrong.")

# =============== 7) 打分：更偏好“修辞触发明显”的样本 ===============
def score_row(r):
    score = 0.0
    # RQ 强
    score += 2.0 * (r.get("rhetorical_question_count", 0) > 0)
    score += 1.0 * (r.get("question_count", 0) > 0)
    # Modality / Hedging
    score += 1.5 * (r.get("hedge_marker_count", 0) > 0)
    score += 1.0 * (r.get("modal_verb_count", 0) > 0)
    score += 1.0 * (r.get("epistemic_strength_score", 0) < 0)  # 更“弱断言”更好解释
    # Contrast
    score += 2.0 * (r.get("has_contrast_structure", 0) == 1)
    score += 1.0 * (r.get("contrast_marker_count", 0) > 0)
    # 文本稍长一点更好解释
    txt = str(r.get(TEXT_COL, ""))
    score += min(len(txt) / 150.0, 1.0)
    return score

if len(err_df) == 0:
    print("No examples found. (This can happen if B3 doesn't hurt on this split.)")
    raise SystemExit

err_df["score"] = err_df.apply(score_row, axis=1)

# =============== 8) 选 2 条：尽量 topic 不同 ===============
err_df = err_df.sort_values("score", ascending=False)

picked = []
used_topics = set()
for _, row in err_df.iterrows():
    t = row.get(TOPIC_COL, "unknown")
    if t in used_topics and len(used_topics) < 2:
        continue
    picked.append(row)
    used_topics.add(t)
    if len(picked) == 2:
        break

# =============== 9) 打印成“论文可直接引用”的格式 ===============
def show_example(r, i):
    print("\n" + "="*80)
    print(f"Example {i}")
    print(f"Topic: {r.get(TOPIC_COL)} | Gold: {r.get('gold')} | B0: {r.get('pred_b0')} | B3: {r.get('pred_b3')}")
    print(f"Text: {r.get(TEXT_COL)}")
    print("--- Features ---")
    print(f"RQ: qc={r.get('question_count')} rc={r.get('rhetorical_question_count')} ratio={r.get('rhetorical_question_ratio')}")
    print(f"MOD: modal={r.get('modal_verb_count')} hedge={r.get('hedge_marker_count')} strong={r.get('strong_assertion_count')} epi={r.get('epistemic_strength_score')}")
    print(f"CON: marker={r.get('contrast_marker_count')} ratio={r.get('contrast_sentence_ratio')} has={r.get('has_contrast_structure')}")
    print("="*80)

for i, r in enumerate(picked, 1):
    show_example(r, i)
'''
C:\Users\27981\PycharmProjects\lichang\.venv\Scripts\python.exe C:\Users\27981\PycharmProjects\lichang\pick_error_analysis_examples.py 
Found 56 cases where B0 correct but B3 wrong.

================================================================================
Example 1
Topic: atheism | Gold: NONE | B0: NONE | B3: FAVOR
Text: @user so, why doesn't a hot air balloon pilot look down to horizon as he rises, rather than horizon being ALWAYS eye level? #SemST
--- Features ---
RQ: qc=1.0 rc=1.0 ratio=1.0
MOD: modal=0.0 hedge=1.0 strong=1.0 epi=0.0
CON: marker=1 ratio=0.5 has=1
================================================================================

================================================================================
Example 2
Topic: hillary | Gold: NONE | B0: NONE | B3: FAVOR
Text: RT @user Live today like it's your last day. But pay bills and dress appropriately just in case it isn't. #420 #SemST
--- Features ---
RQ: qc=0.0 rc=0.0 ratio=0.0
MOD: modal=2.0 hedge=1.0 strong=0.0 epi=-0.3333
CON: marker=1 ratio=0.3333333333333333 has=1
================================================================================

进程已结束，退出代码为 0
'''