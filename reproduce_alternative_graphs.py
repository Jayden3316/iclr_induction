"""reproduce_alternative_graphs.py

Reproduces the accuracy, class-mean PCA, and bigram PCA visualizations for
alternative graph topologies: a Torus graph and a 4-Cube (Hypercube) graph.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import tqdm

# Import shared utilities from utils.py
from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR,
    set_seed, load_model, get_model_accuracies, get_activations,
    compute_class_means, compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)

DATA_DIR = "results/alternative/data"
PLOTS_DIR = "results/alternative/plots"
N_LOOKBACK = 50


# ── Alternative Graph Structures ───────────────────────────────────────────────

class TorusGraph:
    """A 4x4 grid topology with periodic (wrapped) boundary conditions."""
    def __init__(self, words=WORDS, rows=4, cols=4):
        if rows * cols != len(words):
            raise ValueError(f"Dimensions do not match word count.")
        self.words = words
        self.rows = rows
        self.cols = cols
        self.grid = np.array(words).reshape(rows, cols).tolist()
        self.word_to_row = {w: i // cols for i, w in enumerate(words)}
        self.word_to_col = {w: i % cols for i, w in enumerate(words)}

    def generate_sequence(self, seq_len, start_word=None):
        if start_word is not None:
            row, col = self.word_to_row[start_word], self.word_to_col[start_word]
        else:
            row, col = np.random.randint(0, self.rows), np.random.randint(0, self.cols)

        sequence = [self.grid[row][col]]
        while len(sequence) < seq_len:
            # On a torus, all 4 directional movements are always valid
            move = np.random.choice(["up", "down", "left", "right"])
            if move == "up":     row = (row - 1) % self.rows
            elif move == "down":   row = (row + 1) % self.rows
            elif move == "left":   col = (col - 1) % self.cols
            elif move == "right":  col = (col + 1) % self.cols
            sequence.append(self.grid[row][col])
        return sequence

    def generate_batch(self, seq_len):
        return [self.generate_sequence(seq_len, start_word=w) for w in self.words]

    def get_valid_next_words(self, word):
        row, col = self.word_to_row[word], self.word_to_col[word]
        return [
            self.grid[(row - 1) % self.rows][col],
            self.grid[(row + 1) % self.rows][col],
            self.grid[row][(col - 1) % self.cols],
            self.grid[row][(col + 1) % self.cols]
        ]

    def build_adjacency_matrix(self):
        n = len(self.words)
        A = np.zeros((n, n))
        for i, word in enumerate(self.words):
            for neighbor in self.get_valid_next_words(word):
                j = self.words.index(neighbor)
                A[i, j] = 1
        return A


class CubeGraph:
    """A 4-Dimensional Hypercube (16 vertices, each with degree 4).

    Maps perfectly to the 16 vocabulary words using binary Hamming distance.
    """
    def __init__(self, words=WORDS):
        if len(words) != 16:
            raise ValueError("Hypercube graph requires exactly 16 words.")
        self.words = words
        self.word_to_idx = {w: i for i, w in enumerate(words)}

    def _get_neighbors(self, idx):
        # Neighbors share an edge if their 4-bit indices differ by exactly 1 bit
        return [idx ^ (1 << bit) for bit in range(4)]

    def generate_sequence(self, seq_len, start_word=None):
        if start_word is not None:
            idx = self.word_to_idx[start_word]
        else:
            idx = np.random.randint(0, 16)

        sequence = [self.words[idx]]
        while len(sequence) < seq_len:
            neighbors = self._get_neighbors(idx)
            idx = np.random.choice(neighbors)
            sequence.append(self.words[idx])
        return sequence

    def generate_batch(self, seq_len):
        return [self.generate_sequence(seq_len, start_word=w) for w in self.words]

    def get_valid_next_words(self, word):
        idx = self.word_to_idx[word]
        return [self.words[n_idx] for n_idx in self._get_neighbors(idx)]

    def build_adjacency_matrix(self):
        n = len(self.words)
        A = np.zeros((n, n))
        for i, word in enumerate(self.words):
            for neighbor in self.get_valid_next_words(word):
                j = self.word_to_idx[neighbor]
                A[i, j] = 1
        return A


# ── Generalized Plotting Framework ───────────────────────────────────────────

def plot_accuracy_curve(all_accs, graph_name):
    mean = smooth(all_accs.mean(axis=0))
    std = smooth(all_accs.std(axis=0))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(mean, color="black", linewidth=1.0, label="Mean accuracy")
    ax.fill_between(range(len(mean)), mean - std, mean + std,
                    alpha=0.15, color="gray", edgecolor="none", label="$\\pm$1 std")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Sequence Length ({graph_name.capitalize()})", fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray")

    save_figure(fig, PLOTS_DIR, f"{graph_name}_accuracy_curve.pdf")

    # Interactive plotly variant
    pfig = go.Figure()
    x = list(range(len(mean)))
    pfig.add_trace(go.Scatter(x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    pfig.add_trace(go.Scatter(x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(128,128,128,0.2)", showlegend=False, hoverinfo="skip"))
    pfig.add_trace(go.Scatter(x=x, y=mean.tolist(), mode="lines", line=dict(color="black", width=2), name="Mean accuracy", hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>"))
    pfig.update_layout(**plotly_line_layout(f"Accuracy vs Sequence Length ({graph_name.capitalize()})", "Sequence length", "Accuracy"))
    save_plotly(pfig, PLOTS_DIR, f"{graph_name}_accuracy_curve.html")


def plot_class_mean_pca(graph, class_means, pca_dirs, graph_name):
    projected = class_means @ pca_dirs.T

    fig, ax = plt.subplots(figsize=(5, 5))
    A = graph.build_adjacency_matrix()
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
    ax.set_title(f"PCA of Per-Node Mean Activations ({graph_name.capitalize()})", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, f"{graph_name}_pca_class_means.pdf")

    # Interactive plotly variant
    pfig = go.Figure(data=plotly_pca_traces(projected, graph))
    pfig.update_layout(**plotly_pca_layout(f"PCA of Per-Node Mean Activations ({graph_name.capitalize()})"))
    save_plotly(pfig, PLOTS_DIR, f"{graph_name}_pca_class_means.html")


def _draw_bigram_scatter(ax, projected_all, projected_means, tail, graph, label=True):
    A = graph.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected_means[i, 0].item(), projected_means[j, 0].item()],
                    [projected_means[i, 1].item(), projected_means[j, 1].item()],
                    color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                )

    # Geometry paths for split half-circle markers
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


def plot_bigram_pca(graph, sequence, activations, class_means, pca_dirs, graph_name):
    tail = sequence[-N_LOOKBACK:]
    projected_all = activations @ pca_dirs.T
    projected_means = class_means @ pca_dirs.T

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_bigram_scatter(ax, projected_all, projected_means, tail, graph)
    _make_bigram_legend(ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of Individual Activations by Bigram ({graph_name.capitalize()})", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, f"{graph_name}_bigram_pca.pdf")

    # Interactive plotly variant
    pfig = go.Figure(data=plotly_pca_traces(projected_means, graph))
    for word in WORDS:
        idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
        if not idxs:
            continue
        x_coords = [projected_all[idx, 0].item() for idx in idxs]
        y_coords = [projected_all[idx, 1].item() for idx in idxs]
        prev_colors = [WORD_TO_COLOR[tail[idx - 1]] for idx in idxs]

        pfig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="markers", marker=dict(symbol="circle-left", size=10, color=prev_colors), showlegend=False, hoverinfo="skip"))
        pfig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="markers", marker=dict(symbol="circle-right", size=10, color=WORD_TO_COLOR[word]), customdata=[[tail[idx - 1]] for idx in idxs], hovertemplate="current: <b>" + word + "</b><br>previous: <b>%{customdata[0]}</b><extra></extra>", hoverlabel=dict(bgcolor=WORD_TO_COLOR[word], font=dict(color="black")), name=word, showlegend=False))

    pfig.update_layout(**plotly_pca_layout(f"PCA of Individual Activations by Bigram ({graph_name.capitalize()})"))
    pfig.update_layout(width=1000, height=1000)
    pfig.add_annotation(text="◐ Left Half = previous token<br>◑ Right Half = current token<br>★ Token centroid", xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, font=dict(size=12), align="left", bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1, borderpad=6)
    save_plotly(pfig, PLOTS_DIR, f"{graph_name}_bigram_pca.html")


# ── Pipeline Execution Execution ──────────────────────────────────────────────

def run_experiment_pipeline(graph, graph_name, model):
    """Executes caching, transformation processing, and plotting for a given graph object."""
    print(f"\n⚡ Starting analysis pipeline for graph topology: {graph_name.upper()}")
    
    acc_path = os.path.join(DATA_DIR, f"{graph_name}_accuracies.npz")
    pca_path = os.path.join(DATA_DIR, f"{graph_name}_pca.npz")
    seq_path = os.path.join(DATA_DIR, f"{graph_name}_sequence.json")

    if os.path.exists(acc_path) and os.path.exists(pca_path) and os.path.exists(seq_path):
        print(f"Loading cached {graph_name} dataset...")
        all_accs = np.load(acc_path)["all_accs"]
        pca_data = np.load(pca_path)
        activations = pca_data["activations"]
        class_means = pca_data["class_means"]
        pca_dirs = pca_data["pca_dirs"]
        with open(seq_path) as f:
            sequence = json.load(f)
    else:
        os.makedirs(DATA_DIR, exist_ok=True)

        # 1. Compute accuracy metrics
        set_seed(42)
        sequences = graph.generate_batch(SEQ_LEN)
        all_accs = []
        for seq in tqdm.tqdm(sequences, desc=f"Calculating {graph_name} accuracies"):
            all_accs.append(get_model_accuracies(model, graph, seq))
        all_accs = np.array(all_accs)
        np.savez(acc_path, all_accs=all_accs)

        # 2. Extract activation structures and singular vectors (PCA)
        set_seed(42)
        sequence = graph.generate_sequence(SEQ_LEN)
        activations_t = get_activations(model, sequence, LAYER, N_LOOKBACK)
        class_means_t = compute_class_means(activations_t, sequence, WORDS, N_LOOKBACK)
        pca_dirs_t = compute_pca_directions(class_means_t, top_n=2)

        activations = activations_t.cpu().numpy()
        class_means = class_means_t.cpu().numpy()
        pca_dirs = pca_dirs_t.cpu().numpy()

        np.savez(pca_path, activations=activations, class_means=class_means, pca_dirs=pca_dirs)
        with open(seq_path, "w") as f:
            json.dump(sequence, f)

    # 3. Generate and export diagnostic plots
    plot_accuracy_curve(all_accs, graph_name)
    plot_class_mean_pca(graph, class_means, pca_dirs, graph_name)
    plot_bigram_pca(graph, sequence, activations, class_means, pca_dirs, graph_name)
    print(f"✓ Diagnostic plots exported for {graph_name} configuration.")


def main():
    setup_plotting()
    
    # Pre-load shared transformer weight space
    model = load_model()

    # Instantiate graph architectures
    torus_graph = TorusGraph()
    cube_graph = CubeGraph()

    # Run downstream pipeline processing loops
    run_experiment_pipeline(torus_graph, "torus", model)
    run_experiment_pipeline(cube_graph, "cube", model)


if __name__ == "__main__":
    main()