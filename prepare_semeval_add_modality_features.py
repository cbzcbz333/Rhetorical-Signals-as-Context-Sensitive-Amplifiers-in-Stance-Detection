# prepare_semeval_add_modality_features.py

import os
import pandas as pd
from tqdm import tqdm

from feature_extractor import OptimizedFeatureExtractor

# =========================
# 1. 路径配置
# =========================

DATA_PATH = "data/semeval2016_with_rhetorical_features1219add.csv"
OUTPUT_PATH = "data/semeval2016_with_rhetorical_features1219add1.csv"   # 直接覆盖写回（安全）

SAVE_EVERY = 100

MODAL_COLS = [
    "modal_verb_count",
    "hedge_marker_count",
    "strong_assertion_count",
    "epistemic_strength_score"
]

# =========================
# 2. 加载已有数据（第 1 类已完成）
# =========================

df = pd.read_csv(DATA_PATH)
print(f"📊 已加载数据: {len(df)} 条")

# =========================
# 3. 初始化第 2 类特征列（不影响第 1 类）
# =========================

for col in MODAL_COLS:
    if col not in df.columns:
        df[col] = pd.NA

unfinished_mask = df["modal_verb_count"].isna()
unfinished_indices = df[unfinished_mask].index.tolist()

print(f"⏳ 第 2 类尚未处理: {len(unfinished_indices)}")
print(f"✅ 第 2 类已完成: {len(df) - len(unfinished_indices)}")

# =========================
# 4. 初始化 extractor
# =========================

extractor = OptimizedFeatureExtractor(cache_enabled=True)

# =========================
# 5. 只跑第 2 类（支持断点续跑）
# =========================

processed = 0
print("🧠 开始增量提取【模态 / 模糊表达】特征...")

for idx in tqdm(unfinished_indices):
    text = str(df.at[idx, "text"])

    try:
        feats = extractor.extract_features(
            text,
            feature_type="modality"
        )

        df.at[idx, "modal_verb_count"] = feats["modal_verb_count"]
        df.at[idx, "hedge_marker_count"] = feats["hedge_marker_count"]
        df.at[idx, "strong_assertion_count"] = feats["strong_assertion_count"]
        df.at[idx, "epistemic_strength_score"] = feats["epistemic_strength_score"]

    except Exception as e:
        print(f"❌ 第 {idx} 条失败，跳过: {e}")
        continue

    processed += 1

    if processed % SAVE_EVERY == 0:
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"💾 已保存进度（新增 {processed} 条）")

# =========================
# 6. 合并写回 rhetorical_features（关键：不覆盖第 1 类）
# =========================

for idx in df.index:
    base = df.at[idx, "rhetorical_features"]

    if isinstance(base, str):
        base = eval(base)  # CSV 里是字符串字典

    if not isinstance(base, dict):
        base = {}

    # 安全读取第 2 类
    mv = df.at[idx, "modal_verb_count"]
    hm = df.at[idx, "hedge_marker_count"]
    sa = df.at[idx, "strong_assertion_count"]
    es = df.at[idx, "epistemic_strength_score"]

    base["modality_hedging"] = {
        "modal_verb_count": int(mv) if pd.notna(mv) else 0,
        "hedge_marker_count": int(hm) if pd.notna(hm) else 0,
        "strong_assertion_count": int(sa) if pd.notna(sa) else 0,
        "epistemic_strength_score": float(es) if pd.notna(es) else 0.0
    }

    df.at[idx, "rhetorical_features"] = base

# =========================
# 7. 最终保存
# =========================

df.to_csv(OUTPUT_PATH, index=False)
print("\n✅ 第 2 类修辞已成功增量接入（未影响第 1 类）")
