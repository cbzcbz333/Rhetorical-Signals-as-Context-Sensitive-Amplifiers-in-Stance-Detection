# make_rhetoric_density_scatter.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # === 1. 读入你已有的特征表 ===
    # 需要包含列：
    # topic, rhetorical_question_ratio, contrast_sentence_ratio, epistemic_strength_score
    df = pd.read_csv("..\data\semeval2016_with_rhetorical_features1219addcon.csv")

    # === 2. 计算 topic-level rhetorical density ===
    df["mod_abs"] = df["epistemic_strength_score"].abs()

    topic_density = (
        df.groupby("topic")[[
            "rhetorical_question_ratio",
            "contrast_sentence_ratio",
            "mod_abs"
        ]]
        .mean()
        .sum(axis=1)
        .reset_index(name="rhetorical_density")
    )

    # === 3. 手动填入 BERT 的 ΔMacro-F1（N2 − N1） ===
    # 来自你已经算好的 per-topic 结果
    delta_f1 = {
        "abortion": -0.0171,
        "atheism": -0.0767,
        "climate":  0.0365,
        "feminist": -0.0565,
        "hillary": -0.0400,
    }

    topic_density["delta_f1"] = topic_density["topic"].map(delta_f1)

    # === 4. 作图 ===
    fig, ax = plt.subplots(figsize=(6.5, 5))

    x = topic_density["rhetorical_density"]
    y = topic_density["delta_f1"]

    ax.scatter(x, y, s=80)

    # 标注 topic
    for _, row in topic_density.iterrows():
        ax.text(
            row["rhetorical_density"],
            row["delta_f1"],
            row["topic"],
            fontsize=10,
            ha="left",
            va="bottom"
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Topic-level Rhetorical Density")
    ax.set_ylabel("Δ Macro-F1 (BERT + RF − BERT)")
    ax.set_title("Rhetorical Density vs. Performance Gain")

    plt.tight_layout()
    plt.savefig("rhetorical_density_vs_delta_f1.png", dpi=300)
    plt.savefig("rhetorical_density_vs_delta_f1.pdf")
    plt.show()

if __name__ == "__main__":
    main()
