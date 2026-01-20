# baseline_rhetorical_local.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib

# =========================
# 1. 加载数据（你之前整理好的）
# =========================

DATA_PATH = "semeval2016_stance_merged.csv"  # ←你本地路径
OUTPUT_PATH = "semeval2016_with_rhetorical_features.csv"

df = pd.read_csv(DATA_PATH)

# 只用 train / test（如你已有 split 字段）
df_train = df[df["split"] == "train"].copy()
df_test = df[df["split"] == "test"].copy()

X_train_text = df_train["text"].astype(str)
X_test_text = df_test["text"].astype(str)

y_train = df_train["stance"]
y_test = df_test["stance"]

# =========================
# 2. 提取修辞特征（已存在于 DataFrame）
# =========================

def extract_rhetorical_features(df: pd.DataFrame) -> np.ndarray:
    """
    从 DataFrame 中抽取反问修辞特征
    输出 shape: (n_samples, 3)
    """
    return np.vstack([
        df["rhetorical_question_count"].fillna(0).values,
        df["question_count"].fillna(0).values,
        df["rhetorical_question_ratio"].fillna(0.0).values
    ]).T


X_train_rhet = extract_rhetorical_features(df_train)
X_test_rhet = extract_rhetorical_features(df_test)

# =========================
# 3. 文本 Baseline（TF-IDF）
# =========================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=20000
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# =========================
# 4. 特征拼接（文本 + 修辞）
# =========================

X_train_all = hstack([X_train_tfidf, X_train_rhet])
X_test_all = hstack([X_test_tfidf, X_test_rhet])

# =========================
# 5. 训练分类器
# =========================

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=4
)

clf.fit(X_train_all, y_train)

# =========================
# 6. 评估
# =========================

y_pred = clf.predict(X_test_all)

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, digits=4))

macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro-F1: {macro_f1:.4f}")

# =========================
# 7. 保存模型与向量器（可复现实验）
# =========================

joblib.dump(tfidf, "tfidf_baseline.joblib")
joblib.dump(clf, "logreg_rhetorical_baseline.joblib")

# =========================
# 8. 把特征写回 DataFrame（关键步骤）
# =========================

df["rhetorical_features"] = None

for idx in df.index:
    df.at[idx, "rhetorical_features"] = {
        "rhetorical_question": {
            "question_count": int(df.at[idx, "question_count"]),
            "rhetorical_question_count": int(df.at[idx, "rhetorical_question_count"]),
            "rhetorical_question_ratio": float(df.at[idx, "rhetorical_question_ratio"])
        }
    }

df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ 特征已写回数据集并保存到: {OUTPUT_PATH}")
