# make_density_vs_delta_train_dev.py
import json
import numpy as np
import matplotlib.pyplot as plt

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_rf_map(rf_path):
    """
    rf jsonl line example:
    {"id": "7", "rf": [ ... 10 dims ... ]}
    """
    rf_map = {}
    for obj in read_jsonl(rf_path):
        rid = str(obj["id"])
        rf = obj["rf"]
        rf_map[rid] = rf
    return rf_map

def topic_density_from_train_dev(train_path, dev_path, rf_path):
    """
    Using ONLY train+dev to compute topic-level rhetorical density:
      density_i = rq_ratio + contrast_sentence_ratio + abs(epistemic_strength_score)

    rf vector layout (your current 10-dim):
      0 question_count
      1 rhetorical_question_count
      2 rhetorical_question_ratio
      3 modal_verb_count
      4 hedge_marker_count
      5 strong_assertion_count
      6 epistemic_strength_score
      7 contrast_marker_count
      8 contrast_sentence_ratio
      9 has_contrast_structure
    """
    rf_map = build_rf_map(rf_path)

    samples = []
    for path in [train_path, dev_path]:
        for obj in read_jsonl(path):
            rid = str(obj["id"])
            topic = str(obj.get("topic", "")).lower()
            if rid not in rf_map:
                continue
            rf = rf_map[rid]
            # Defensive: ensure length >= 9
            if not isinstance(rf, list) or len(rf) < 9:
                continue

            rq_ratio = float(rf[2])
            mod_abs = abs(float(rf[6]))
            con_ratio = float(rf[8])

            density = rq_ratio + con_ratio + mod_abs
            samples.append((topic, density))

    # Aggregate mean density per topic
    topic2vals = {}
    for topic, density in samples:
        topic2vals.setdefault(topic, []).append(density)

    topic_stats = []
    for topic, vals in topic2vals.items():
        topic_stats.append((topic, float(np.mean(vals)), len(vals)))

    # Sort by density (optional; makes reading easier)
    topic_stats.sort(key=lambda x: x[1])
    return topic_stats

def main():
    # ====== paths (edit these) ======
    train_path = "../data/train.jsonl"
    dev_path = "../data/validation.jsonl"  # you called it validation
    rf_path = "../data/rhetoric_features_all0113addcon2.jsonl"

    # ====== your ΔMacro-F1 on TEST (fixed numbers from your results/tables) ======
    delta_bert = {
        "abortion": -0.0171,
        "atheism": -0.0767,
        "climate":  0.0365,
        "feminist": -0.0565,
        "hillary": -0.0400,
    }

    delta_lr = {
        "abortion": -0.0432,
        "atheism": -0.0246,
        "climate":  -0.0395,
        "feminist": -0.0077,
        "hillary":   0.0106,
    }

    # ====== compute train+dev densities ======
    stats = topic_density_from_train_dev(train_path, dev_path, rf_path)
    # stats: [(topic, mean_density, n), ...]

    # Keep only topics that exist in both delta dicts
    rows = []
    for topic, dens, n in stats:
        if topic in delta_bert and topic in delta_lr:
            rows.append((topic, dens, n, delta_lr[topic], delta_bert[topic]))

    if not rows:
        raise RuntimeError("No matching topics found. Check topic names in jsonl (lowercase?) and delta dict keys.")

    # Prepare arrays
    topics = [r[0] for r in rows]
    x = np.array([r[1] for r in rows], dtype=float)
    y_lr = np.array([r[3] for r in rows], dtype=float)
    y_bert = np.array([r[4] for r in rows], dtype=float)

    # ====== plot (single figure) ======
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.scatter(x, y_lr, s=90, marker="o", label="TF–IDF+LR  Δ(B3−B0)")
    ax.scatter(x, y_bert, s=90, marker="^", label="BERT  Δ(N2−N1)")

    # annotate topic near BERT point
    #for t, xi, yi in zip(topics, x, y_bert):
        #ax.text(xi, yi, t, fontsize=10, ha="left", va="bottom")
    '''
    for t, xi, y_lr_i, y_bert_i in zip(topics, x, y_lr, y_bert):
        y_mid = (y_lr_i + y_bert_i) / 2
        ax.text(
            xi,
            y_mid,
            t,
            fontsize=10,
            ha="center",
            va="center"
        )
    
    for t, xi, yi in zip(topics, x, y_lr):
        ax.text(xi, yi, t, fontsize=9, ha="right", va="top")

    for t, xi, yi in zip(topics, x, y_bert):
        ax.text(xi, yi, t, fontsize=9, ha="left", va="bottom")
    '''
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("Topic-level Rhetorical Density (train+dev)")
    ax.set_ylabel("Δ Macro-F1 on test (Rhetorical − Text-only)")
    ax.set_title("Rhetorical Density vs. Performance Gain (No Test Leakage)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("density_vs_delta_train_dev.png", dpi=600)
    plt.savefig("density_vs_delta_train_dev.pdf")
    plt.show()

    # print a small table for sanity
    print("\nTopic  n(train+dev)  density(train+dev)  ΔLR(test)  ΔBERT(test)")
    for t, dens, n, dlr, dbert in rows:
        print(f"{t:8s} {n:10d} {dens:17.4f} {dlr:9.4f} {dbert:10.4f}")

if __name__ == "__main__":
    main()
