# make_delta_f1_heatmap_paper.py
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Raw data
    topics = ["Abortion", "Atheism", "Climate", "Feminist", "Hillary"]

    delta_lr   = np.array([-0.0432, -0.0246, -0.0395, -0.0077,  0.0106])
    delta_bert = np.array([-0.0171, -0.0767,  0.0365, -0.0565, -0.0400])

    # Sort topics by BERT delta (ascending -> descending)
    order = np.argsort(delta_bert)
    topics = [topics[i] for i in order]
    delta_lr = delta_lr[order]
    delta_bert = delta_bert[order]

    data = np.vstack([delta_lr, delta_bert])
    row_labels = ["LR Δ(B3−B0)", "BERT Δ(N2−N1)"]

    # Symmetric color scale around 0
    vmax = np.max(np.abs(data))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10, 3.0))

    im = ax.imshow(
        data,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm"
    )

    # Ticks & labels
    ax.set_xticks(np.arange(len(topics)))
    ax.set_xticklabels(topics)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j, i,
                f"{data[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=10
            )

    # Grid lines
    ax.set_xticks(np.arange(-.5, len(topics), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Macro-F1 (Rhetorical − Text-only)")

    ax.set_title("Topic-Conditional Effects of Rhetorical Features")

    plt.tight_layout()
    plt.savefig("delta_f1_heatmap_paper.png", dpi=300)
    plt.savefig("delta_f1_heatmap_paper.pdf")
    plt.show()

if __name__ == "__main__":
    main()
