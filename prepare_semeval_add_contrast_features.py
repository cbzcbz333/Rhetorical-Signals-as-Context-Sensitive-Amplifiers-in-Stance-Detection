# prepare_semeval_add_modality_features.py

import os
import pandas as pd
from tqdm import tqdm

from feature_extractor import OptimizedFeatureExtractor
from contrast_extractor import extract_contrast_features

# =========================
# 1. 路径配置
# =========================

DATA_PATH = "data/semeval2016_with_rhetorical_features1219add1.csv"
OUTPUT_PATH = "data/semeval2016_with_rhetorical_features1219addcon.csv"   # 直接覆盖写回（安全）

SAVE_EVERY = 100

CONTRAST_COLS = [
    "contrast_marker_count",
    "contrast_sentence_ratio",
    "has_contrast_structure"
]

# =========================
# 2. 加载已有数据（第 1 & 2类已完成）
# =========================

df = pd.read_csv(DATA_PATH)
print(f"📊 已加载数据: {len(df)} 条")

# =========================
# 3. 初始化第 3 类特征列（不影响1 & 2类）
# =========================

for col in CONTRAST_COLS:
    if col not in df.columns:
        df[col] = pd.NA

unfinished_mask = df["contrast_marker_count"].isna()
unfinished_indices = df[unfinished_mask].index.tolist()

print(f"⏳ 第 3 类尚未处理: {len(unfinished_indices)}")
print(f"✅ 第 3 类已完成: {len(df) - len(unfinished_indices)}")

# =========================
# 4. 初始化 extractor
# =========================

#extractor = extract_contrast_features(cache_enabled=True)

# =========================
# 5. 只跑第 3 类（支持断点续跑）
# =========================

processed = 0
print("🧠 开始增量提取【模态 / 模糊表达】特征...")

for idx in tqdm(unfinished_indices):
    text = str(df.at[idx, "text"])

    try:
        """
        feats = extractor.extract_features(
            text,
            feature_type="modality"
        )#"""
        feats = extract_contrast_features(text)

        df.at[idx, "contrast_marker_count"] = feats["contrast_marker_count"]
        df.at[idx, "contrast_sentence_ratio"] = feats["contrast_sentence_ratio"]
        df.at[idx, "has_contrast_structure"] = feats["has_contrast_structure"]

    except Exception as e:
        print(f"❌ 第 {idx} 条失败，跳过: {e}")
        continue

    processed += 1

    if processed % SAVE_EVERY == 0:
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"💾 已保存进度（新增 {processed} 条）")

# =========================
# 6. 合并写回 rhetorical_features（关键：不覆盖第 1 & 2类）
# =========================

for idx in df.index:
    base = df.at[idx, "rhetorical_features"]

    if isinstance(base, str):
        base = eval(base)  # CSV 里是字符串字典

    if not isinstance(base, dict):
        base = {}

    # 安全读取第 3 类
    cmc = df.at[idx, "contrast_marker_count"]
    csr = df.at[idx, "contrast_sentence_ratio"]
    hcs = df.at[idx, "has_contrast_structure"]

    base["contrast_opposition"] = {
        "contrast_marker_count": int(cmc) if pd.notna(cmc) else 0,
        "contrast_sentence_ratio": float(csr) if pd.notna(csr) else 0,
        "has_contrast_structure": int(hcs) if pd.notna(hcs) else 0,
    }

    df.at[idx, "rhetorical_features"] = base

# =========================
# 7. 最终保存
# =========================

df.to_csv(OUTPUT_PATH, index=False)
print("\n✅ 第 3 类修辞已成功增量接入（未影响第 1 & 2 类）")
