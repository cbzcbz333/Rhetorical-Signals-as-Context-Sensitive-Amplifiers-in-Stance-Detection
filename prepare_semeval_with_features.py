# prepare_semeval_with_features.py

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ⚠️ 确保 feature_extractor.py 在同一目录
from feature_extractor import OptimizedFeatureExtractor

# =========================
# 1. 基本配置
# =========================

DATA_DIR = "data"
#RAW_PATH = os.path.join(DATA_DIR, "semeval2016_raw.csv")
RAW_PATH = os.path.join(DATA_DIR, "semeval2016_with_rhetorical_features.csv")


OUTPUT_PATH = os.path.join(DATA_DIR, "semeval2016_with_rhetorical_features1219.csv")

STANCE_TARGETS = [
    "stance_atheism",
    "stance_feminist",
    "stance_hillary",
    "stance_abortion",
    "stance_climate"
]

LABEL_MAP = {
    0: "NONE",
    1: "AGAINST",
    2: "FAVOR"
}

os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# 2. 下载并合并 SemEval-2016
# =========================

def load_and_merge_semeval() -> pd.DataFrame:
    dfs = []

    print("📥 Loading SemEval-2016 (tweet_eval)...")

    for target in STANCE_TARGETS:
        topic = target.replace("stance_", "")
        ds = load_dataset("tweet_eval", target)

        for split in ["train", "validation", "test"]:
            df = ds[split].to_pandas()
            df["topic"] = topic
            df["split"] = split
            df["stance"] = df["label"].map(LABEL_MAP)
            dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df = merged_df[
        ["text", "topic", "stance", "split"]
    ]

    return merged_df


# =========================
# 3. 加载或生成原始数据
# =========================

if os.path.exists(RAW_PATH):
    print(f"✅ 使用已存在数据: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
else:
    df = load_and_merge_semeval()
    df.to_csv(RAW_PATH, index=False)
    print(f"✅ 原始数据已保存: {RAW_PATH}")

print(f"📊 数据规模: {len(df)}")

# =========================
# 4. 初始化反问修辞特征提取器
# =========================

extractor = OptimizedFeatureExtractor(cache_enabled=True)
SAVE_EVERY = 200          # 每 200 条保存一次
CSV_FLUSH_MODE = "w"     # 覆盖写（安全）
# 初始化特征列（避免重复跑）
'''
if "question_count" not in df.columns:
    df["question_count"] = None
    df["rhetorical_question_count"] = None
    df["rhetorical_question_ratio"] = None
'''
FEATURE_COLS = [
    "question_count",
    "rhetorical_question_count",
    "rhetorical_question_ratio"
]

for col in FEATURE_COLS:
    if col not in df.columns:
        df[col] = pd.NA
# =========================
# 识别未完成样本（断点续跑核心）
# =========================

unfinished_mask = df["question_count"].isna()
unfinished_indices = df[unfinished_mask].index.tolist()
print(f"📌 总样本数: {len(df)}")
print(f"⏳ 尚未处理: {len(unfinished_indices)}")
print(f"✅ 已完成: {len(df) - len(unfinished_indices)}")

# =========================
# 5. 提取反问修辞特征（一次性）
# =========================

print("🧠 Extracting rhetorical question features...")

'''
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # 跳过已提取的样本（支持断点续跑）
    if pd.notna(row["question_count"]):
        continue

    text = str(row["text"])

    result = extractor.extract_features(text)

    df.at[idx, "question_count"] = result.question_count
    df.at[idx, "rhetorical_question_count"] = result.rhetorical_count
    df.at[idx, "rhetorical_question_ratio"] = result.rhetorical_ratio
'''
SAVE_EVERY = 10
processed_since_save = 0

print("🧠 开始断点续跑式特征提取...")

for idx in tqdm(unfinished_indices):
    text = str(df.at[idx, "text"])

    try:
        result = extractor.extract_features(text)

        df.at[idx, "question_count"] = result.question_count
        df.at[idx, "rhetorical_question_count"] = result.rhetorical_count
        df.at[idx, "rhetorical_question_ratio"] = result.rhetorical_ratio

    except Exception as e:
        print(f"❌ 第 {idx} 条失败，跳过: {e}")
        continue

    processed_since_save += 1

    # ====== 每 SAVE_EVERY 条强制写盘 ======
    if processed_since_save % SAVE_EVERY == 0:
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"💾 已保存进度（最近处理 {processed_since_save} 条）")

# =========================
# 6. 结构化保存（为后续扩展准备）——最终安全版
# =========================

df["rhetorical_features"] = None

for idx in df.index:
    qc = df.at[idx, "question_count"]
    rc = df.at[idx, "rhetorical_question_count"]
    rr = df.at[idx, "rhetorical_question_ratio"]

    # ---- question_count ----
    if isinstance(qc, bool):
        qc = int(qc)
    elif isinstance(qc, (int, float)) and pd.notna(qc):
        qc = int(qc)
    else:
        qc = 0

    # ---- rhetorical_question_count ----
    if isinstance(rc, bool):
        rc = int(rc)
    elif isinstance(rc, (int, float)) and pd.notna(rc):
        rc = int(rc)
    else:
        rc = 0

    # ---- rhetorical_question_ratio ----
    if isinstance(rr, (int, float)) and pd.notna(rr):
        rr = float(rr)
    else:
        rr = 0.0

    df.at[idx, "rhetorical_features"] = {
        "rhetorical_question": {
            "question_count": qc,
            "rhetorical_question_count": rc,
            "rhetorical_question_ratio": rr
        }
    }

df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ 完成！特征化数据已保存至:\n{OUTPUT_PATH}")

# =========================
# 7. 简要统计（ sanity check ）
# =========================

print("\n📈 Rhetorical Question Feature Stats:")
print(df[[
    "question_count",
    "rhetorical_question_count",
    "rhetorical_question_ratio"
]].describe())
