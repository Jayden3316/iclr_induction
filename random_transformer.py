"""random_transformer.py

Reproduces the grid graph visualizations (accuracy curve, class-mean PCA, 
and bigram PCA) using the Llama-3.1-8B architecture initialized with 
completely random weights.
"""

import os
import json
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import tqdm

from transformer_lens import HookedTransformer

# Import shared utilities and constants from utils.py
from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR,
    Grid, set_seed, load_model, get_model_accuracies, get_activations,
    compute_class_means, compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)

DATA_DIR = "results/random/data"
PLOTS_DIR = "results/random/plots"
N_LOOKBACK = 50


# ── Random Model Initialization ───────────────────────────────────────────────

def load_random_model():
    """Loads the architecture configuration and tokenizer of Llama, 

    but instantiates it with completely randomized weights.
    """
    print("Fetching architecture config and tokenizer from target model...")
    pretrained = load_model()
    cfg = pretrained.cfg
    tokenizer = pretrained.tokenizer
    
    # Safely purge pretrained parameters from RAM/VRAM
    del pretrained
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Instantiating random weight space for the architecture...")
    # Instantiating directly via HookedTransformer triggers default random init
    model = HookedTransformer(cfg)
    model.tokenizer = tokenizer
    return model


# ── Plotting Implementations ──────────────────────────────────────────────────

def plot_accuracy_curve(all_accs):
    mean = smooth(all_accs.mean(axis=0))
    std = smooth(all_accs.std(axis=0))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(mean, color="black", linewidth=1.0, label="Mean accuracy")
    ax.fill_between(range(len(mean)), mean - std, mean + std,
                    alpha=0.15, color="gray", edgecolor="none", label="±1 std")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy (grid task)")
    ax.set_title("Accuracy vs Sequence Length (Random Weights)", fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray")

    save_figure(fig, PLOTS_DIR, "accuracy_curve_random.pdf")

    # Interactive variant
    pfig = go.Figure()
    x = list(range(len(mean)))
    pfig.add_trace(go.Scatter(x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    pfig.add_trace(go.Scatter(x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(128,128,128,0.2)", showlegend=False, hoverinfo="skip"))
    pfig.add_trace(go.Scatter(x=x, y=mean.tolist(), mode="lines", line=dict(color="black", width=2), name="Mean accuracy", hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>"))
    pfig.update_layout(**plotly_line_layout("Accuracy vs Sequence Length (Random Weights)", "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, PLOTS_DIR, "accuracy_curve_random.html")


def plot_class_mean_pca(grid, class_means, pca_dirs):
    projected = class_means @ pca_dirs.T

    fig, ax = plt.subplots(figsize=(5, 5))
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0].item(), projected[j, 0].item()],
                    [projected[i, 1].item(), projected[j, 1].item()],
                    color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                )

    for i, word in enumerate(WORDS):
        ax.scatter(projected[i, 0].item(), projected[i, 1].item(), color=WORD_TO_COLOR[word], s=120, marker="*", edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(word, (projected[i, 0].item(), projected[i, 1].item()), xytext=(5, 5), textcoords="offset points", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of Per-Node Mean Activations (Random Weights)", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, "pca_class_means_random.pdf")

    # Interactive variant
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout("PCA of Per-Node Mean Activations (Random Weights)"))
    save_plotly(pfig, PLOTS_DIR, "pca_class_means_random.html")


def _draw_bigram_scatter(ax, projected_all, projected_means, tail, grid, label=True):
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected_means[i, 0].item(), projected_means[j, 0].item()],
                    [projected_means[i, 1].item(), projected_means[j, 1].item()],
                    color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                )

    # Clean geometric half-circles for dual-token identity representation
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, 50)
    verts_left = np.column_stack([np.cos(theta_left), np.sin(theta_left)])
    verts_left = np.vstack([verts_left, [0, 0], [0, 1]])
    left_half_circle = mpath.Path(verts_left)

    theta_right = np.linspace(-np.pi/2, np.pi/2, 50)
    verts_right = np.column_stack([np.cos(theta_right), np.sin(theta_right)])
    verts_right = np.vstack([verts_right, [0, 0], [0, -1]])
    right_half_circle = mpath.Path(verts_right)

    for idx in range(1, len(tail)):
        cur_word = tail[idx]
        prev_word = tail[idx - 1]
        x_pos = projected_all[idx, 0].item()
        y_pos = projected_all[idx, 1].item()
        
        ax.scatter(x_pos, y_pos, color=WORD_TO_COLOR[prev_word], marker=left_half_circle, s=50, alpha=1.0, zorder=3)
        ax.scatter(x_pos, y_pos, color=WORD_TO_COLOR[cur_word], marker=right_half_circle, s=50, alpha=1.0, zorder=3)

    for i, word in enumerate(WORDS):
        ax.scatter(projected_means[i, 0].item(), projected_means[i, 1].item(), color=WORD_TO_COLOR[word], s=120, marker="*", edgecolors="black", linewidths=1.0, zorder=5)
        if label:
            ax.annotate(word, (projected_means[i, 0].item(), projected_means[i, 1].item()), xytext=(5, 5), textcoords="offset points", fontsize=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))


def _make_bigram_legend(ax):
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, 50)
    verts_left = np.column_stack([np.cos(theta_left), np.sin(theta_left)])
    verts_left = np.vstack([verts_left, [0, 0], [0, 1]])
    left_half_circle = mpath.Path(verts_left)

    theta_right = np.linspace(-np.pi/2, np.pi/2, 50)
    verts_right = np.column_stack([np.cos(theta_right), np.sin(theta_right)])
    verts_right = np.vstack([verts_right, [0, 0], [0, -1]])
    right_half_circle = mpath.Path(verts_right)

    legend_elements = [
        Line2D([0], [0], marker=left_half_circle, color="w", markerfacecolor="gray", markersize=10, label="Left half = previous token"),
        Line2D([0], [0], marker=right_half_circle, color="w", markerfacecolor="darkgray", markersize=10, label="Right half = current token"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markersize=12, markeredgecolor="black", markeredgewidth=0.8, label="Token centroid"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=8)


def plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs):
    tail = sequence[-N_LOOKBACK:]
    projected_all = activations @ pca_dirs.T
    projected_means = class_means @ pca_dirs.T

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_bigram_scatter(ax, projected_all, projected_means, tail, grid)
    _make_bigram_legend(ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of Individual Activations by Bigram (Random Weights)", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, "bigram_pca_random.pdf")

    # Interactive variant
    pfig = go.Figure(data=plotly_pca_traces(projected_means, grid))
    for word in WORDS:
        idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
        if not idxs:
            continue
            
        x_coords = [projected_all[idx, 0].item() for idx in idxs]
        y_coords = [projected_all[idx, 1].item() for idx in idxs]
        prev_colors = [WORD_TO_COLOR[tail[idx - 1]] for idx in idxs]

        pfig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="markers", marker=dict(symbol="circle-left", size=10, color=prev_colors), showlegend=False, hoverinfo="skip"))
        pfig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="markers", marker=dict(symbol="circle-right", size=10, color=WORD_TO_COLOR[word]), customdata=[[tail[idx - 1]] for idx in idxs], hovertemplate="current: <b>" + word + "</b><br>previous: <b>%{customdata[0]}</b><extra></extra>", hoverlabel=dict(bgcolor=WORD_TO_COLOR[word], font=dict(color="black")), name=word, showlegend=False))

    pfig.update_layout(**plotly_pca_layout("PCA of Individual Activations by Bigram (Random Weights)"))
    pfig.update_layout(width=1000, height=1000)
    pfig.add_annotation(text="◐ Left Half = previous token<br>◑ Right Half = current token<br>★ Token centroid", xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, font=dict(size=12), align="left", bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1, borderpad=6)
    save_plotly(pfig, PLOTS_DIR, "bigram_pca_random.html")


# ── Main Control Pipeline ─────────────────────────────────────────────────────

def main():
    setup_plotting()
    grid = Grid()

    acc_path = os.path.join(DATA_DIR, "random_accuracies.npz")
    pca_path = os.path.join(DATA_DIR, "random_pca.npz")
    seq_path = os.path.join(DATA_DIR, "random_sequence.json")

    if os.path.exists(acc_path) and os.path.exists(pca_path) and os.path.exists(seq_path):
        print("Found cached random-weight dataset. Rendering plots directly...")
        all_accs = np.load(acc_path)["all_accs"]
        pca_data = np.load(pca_path)
        activations = pca_data["activations"]
        class_means = pca_data["class_means"]
        pca_dirs = pca_data["pca_dirs"]
        with open(seq_path) as f:
            sequence = json.load(f)
    else:
        model = load_random_model()
        os.makedirs(DATA_DIR, exist_ok=True)

        # 1. Compute accuracy metrics on the random model (should look uniform/flat)
        set_seed(42)
        sequences = grid.generate_batch(SEQ_LEN)
        all_accs = []
        for seq in tqdm.tqdm(sequences, desc="Evaluating random model accuracy"):
            all_accs.append(get_model_accuracies(model, grid, seq))
        all_accs = np.array(all_accs)
        np.savez(acc_path, all_accs=all_accs)

        # 2. Extract activation structures under random projections
        set_seed(42)
        sequence = grid.generate_sequence(SEQ_LEN)
        activations_t = get_activations(model, sequence, LAYER, N_LOOKBACK)
        class_means_t = compute_class_means(activations_t, sequence, WORDS, N_LOOKBACK)
        pca_dirs_t = compute_pca_directions(class_means_t, top_n=2)

        activations = activations_t.cpu().numpy()
        class_means = class_means_t.cpu().numpy()
        pca_dirs = pca_dirs_t.cpu().numpy()

        np.savez(pca_path, activations=activations, class_means=class_means, pca_dirs=pca_dirs)
        with open(seq_path, "w") as f:
            json.dump(sequence, f)

    # ── Render Outputs ────────────────────────────────────────────────────────
    plot_accuracy_curve(all_accs)
    plot_class_mean_pca(grid, class_means, pca_dirs)
    plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs)
    print("Done! Evaluation artifacts written to execution directory.")


if __name__ == "__main__":
    main()