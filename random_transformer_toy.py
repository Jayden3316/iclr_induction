"""random_transformer_toy.py

Reproduces the grid graph visualizations using a minimal, toy Transformer 
architecture: 1 attention layer, 0 MLP blocks, initialized with completely 
random weights.
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

from transformer_lens import HookedTransformer, HookedTransformerConfig

# Import shared utilities and constants from utils.py
from utils import (
    WORDS, SEQ_LEN, WORD_TO_COLOR,
    Grid, set_seed, load_model, get_model_accuracies, get_activations,
    compute_class_means, compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)

DATA_DIR = "results/toy/data"
PLOTS_DIR = "results/toy/plots"
N_LOOKBACK = 50
TOY_LAYER = 0  # Only layer 0 exists in a 1-layer model


# ── Toy Attention-Only Model Initialization ───────────────────────────────────

from transformers import AutoTokenizer  # Add this import at the top

def load_toy_model():
    """Initializes a 1-layer, attention-only toy HookedTransformer 
    using the Llama tokenizer configuration WITHOUT downloading the 16GB weights.
    """
    print("Fetching tokenizer parameters directly from Hugging Face...")
    # This only downloads a few kilobytes of text configurations
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    vocab_size = len(tokenizer) # Llama 3/3.1 uses a 128,256 token vocabulary
        
    print(f"Building toy model architecture (1 Layer, Attention-Only, Vocab: {vocab_size})...")
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=256,         
        n_heads=8,           
        d_head=32,           
        n_ctx=2048,          
        d_vocab=vocab_size,
        attn_only=True,      
        normalization_type="LN",
        act_fn="silu",
    )
    
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
    ax.set_title("Accuracy vs Sequence Length (Toy Attn-Only Model)", fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray")

    save_figure(fig, PLOTS_DIR, "accuracy_curve_toy.pdf")

    # Interactive variant
    pfig = go.Figure()
    x = list(range(len(mean)))
    pfig.add_trace(go.Scatter(x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    pfig.add_trace(go.Scatter(x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(128,128,128,0.2)", showlegend=False, hoverinfo="skip"))
    pfig.add_trace(go.Scatter(x=x, y=mean.tolist(), mode="lines", line=dict(color="black", width=2), name="Mean accuracy", hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>"))
    pfig.update_layout(**plotly_line_layout("Accuracy vs Sequence Length (Toy Attn-Only Model)", "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, PLOTS_DIR, "accuracy_curve_toy.html")


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
    ax.set_title("PCA of Per-Node Mean Activations (Toy Attn-Only Model)", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, "pca_class_means_toy.pdf")

    # Interactive variant
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout("PCA of Per-Node Mean Activations (Toy Attn-Only Model)"))
    save_plotly(pfig, PLOTS_DIR, "pca_class_means_toy.html")


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

    # Establish clean paths for true geometric half-circles
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
    ax.set_title("PCA of Individual Activations by Bigram (Toy Attn-Only Model)", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, "bigram_pca_toy.pdf")

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

    pfig.update_layout(**plotly_pca_layout("PCA of Individual Activations by Bigram (Toy Attn-Only Model)"))
    pfig.update_layout(width=1000, height=1000)
    pfig.add_annotation(text="◐ Left Half = previous token<br>◑ Right Half = current token<br>★ Token centroid", xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, font=dict(size=12), align="left", bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1, borderpad=6)
    save_plotly(pfig, PLOTS_DIR, "bigram_pca_toy.html")


def plot_attention_matrices(model, sequence):
    """Extracts and plots the attention patterns for all 8 heads in Layer 0

    focused on the last N_LOOKBACK tokens.
    """
    print("Extracting attention patterns from the single layer...")
    
    # Format sequence into a single string for the tokenizer
    text = " ".join(sequence)
    tokens = model.to_tokens(text)
    
    # Run the model and grab the activation cache
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        
    # HookedTransformer attention pattern shape: [batch, head, query_pos, key_pos]
    # Squeeze batch dimension to get [8, seq_len, seq_len]
    attn_pattern = cache["pattern", 0, "attn"][0].cpu().numpy()
    
    # Get the exact string tokens to match matrix dimensions
    str_tokens = model.to_str_tokens(tokens)
    
    # Slice out the final N_LOOKBACK tokens for visualization
    # Note: Attention rows may not sum to 1 in this window if a query 
    # decides to look further back into the historical sequence context.
    vis_tokens = str_tokens[-N_LOOKBACK:]
    attn_slice = attn_pattern[:, -N_LOOKBACK:, -N_LOOKBACK:]
    
    # Create a 2x4 grid subplot layout for the 8 attention heads
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    axes = axes.flatten()
    
    for head in range(model.cfg.n_heads):
        ax = axes[head]
        im = ax.imshow(attn_slice[head], cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Head {head}", fontsize=10)
        
        # Clean up tick clutter; only show labels on boundary edges
        if head >= 4:
            ax.set_xlabel("Key Position (Context)", fontsize=9)
        if head % 4 == 0:
            ax.set_ylabel("Query Position (Current)", fontsize=9)
            
        # Optional: Uncomment if you want to label tick points with words 
        # (Warning: 50 labels can overlap tightly on small displays)
        # ax.set_xticks(range(0, N_LOOKBACK, 5))
        # ax.set_yticks(range(0, N_LOOKBACK, 5))
        
    # Add a unified colorbar for the figure
    fig.subplots_adjust(right=0.90, hspace=0.25, wspace=0.15)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cbar_ax=cbar_ax, label="Attention Probability Weight")
    
    fig.suptitle("Attention Matrices Matrix Layout (Layer 0, Last 50 Tokens)", fontsize=12, y=0.96)
    
    save_figure(fig, PLOTS_DIR, "attention_matrices_toy.pdf")
    plt.close(fig)
    print(f"Attention matrices successfully generated and saved to {PLOTS_DIR}/attention_matrices_toy.pdf")

# ── Execution Pipeline ────────────────────────────────────────────────────────

def main():
    setup_plotting()
    grid = Grid()

    acc_path = os.path.join(DATA_DIR, "toy_accuracies.npz")
    pca_path = os.path.join(DATA_DIR, "toy_pca.npz")
    seq_path = os.path.join(DATA_DIR, "toy_sequence.json")

    if os.path.exists(acc_path) and os.path.exists(pca_path) and os.path.exists(seq_path):
        print("Found cached toy-model dataset. Generating plots directly...")
        all_accs = np.load(acc_path)["all_accs"]
        pca_data = np.load(pca_path)
        activations = pca_data["activations"]
        class_means = pca_data["class_means"]
        pca_dirs = pca_data["pca_dirs"]
        with open(seq_path) as f:
            sequence = json.load(f)
            
        # Re-load the model carcass structure if you need to run attention plots on cache hits
        model = load_toy_model()
    else:
        model = load_toy_model()
        os.makedirs(DATA_DIR, exist_ok=True)

        # 1. Compute baseline accuracies
        set_seed(42)
        sequences = grid.generate_batch(SEQ_LEN)
        all_accs = []
        for seq in tqdm.tqdm(sequences, desc="Evaluating toy model accuracy"):
            all_accs.append(get_model_accuracies(model, grid, seq))
        all_accs = np.array(all_accs)
        np.savez(acc_path, all_accs=all_accs)

        # 2. Collect activations from layer 0 residual stream
        set_seed(42)
        sequence = grid.generate_sequence(SEQ_LEN)
        activations_t = get_activations(model, sequence, TOY_LAYER, N_LOOKBACK)
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
    # plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs)
    
    # Run your new attention matrix plot
    plot_attention_matrices(model, sequence)
    
    print("Execution complete. Visualizations written to results/toy/plots/")

    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import (
        WORDS, WORD_TO_COLOR, Grid, set_seed, get_activations, 
        compute_pca_directions, save_figure
    )

    # Configuration Constants
    PLOTS_DIR = "results/toy/plots"
    N_LOOKBACK = 50
    TOY_LAYER = 0

    # Define your parameter matrix
    seq_lens = [64, 256, 1024]
    n_sequences_list = [4, 16, 64, 256]

    grid = Grid()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for seq_len in seq_lens:
        for n_seq in n_sequences_list:
            print(f"Processing configurations for SEQ_LEN={seq_len} | N_SEQUENCES={n_seq}...")
            
            set_seed(42) # Optional: ensures identical random walks across runs
            
            # Initialize an accumulation dictionary for token signal vectors
            word_signals = {word: [] for word in WORDS}
            
            # 1. Process each sequence within the model's n_ctx boundaries
            for _ in range(n_seq):
                sequence = grid.generate_sequence(seq_len)
                
                # Extract activations for the tail of this specific walk
                # Shape: [N_LOOKBACK, d_model]
                acts_t = get_activations(model, sequence, TOY_LAYER, n_lookback=N_LOOKBACK)
                tail = sequence[-N_LOOKBACK:]
                
                # Map vectors to their actual word classes
                for idx, word in enumerate(tail):
                    word_signals[word].append(acts_t[idx])
            
            # 2. Compute true global class means across all accumulated paths
            class_means_list = []
            for word in WORDS:
                if word_signals[word]:
                    class_means_list.append(torch.stack(word_signals[word]).mean(dim=0))
                else:
                    # Safe fallback if a word never appeared in any sequence tail
                    class_means_list.append(torch.zeros(model.cfg.d_model, device=model.cfg.device))
                    
            class_means_t = torch.stack(class_means_list)
            
            # 3. Calculate PCA directions & project down to 2D
            pca_dirs_t = compute_pca_directions(class_means_t, top_n=2)
            
            class_means = class_means_t.cpu().numpy()
            pca_dirs = pca_dirs_t.cpu().numpy()
            projected = class_means @ pca_dirs.T
            
            # 4. Generate the Visualization Layout
            fig, ax = plt.subplots(figsize=(5, 5))
            A = grid.build_adjacency_matrix()
            
            # Draw structural graph edges
            for i in range(len(WORDS)):
                for j in range(i + 1, len(WORDS)):
                    if A[i, j]:
                        ax.plot(
                            [projected[i, 0].item(), projected[j, 0].item()],
                            [projected[i, 1].item(), projected[j, 1].item()],
                            color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                        )
            
            # Draw representation node centroids
            for i, word in enumerate(WORDS):
                ax.scatter(
                    projected[i, 0].item(), projected[i, 1].item(), 
                    color=WORD_TO_COLOR[word], s=120, marker="*", 
                    edgecolors="black", linewidths=0.5, zorder=5
                )
                ax.annotate(
                    word, (projected[i, 0].item(), projected[i, 1].item()), 
                    xytext=(5, 5), textcoords="offset points", fontsize=8, 
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
                )
                
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA Representation (L={seq_len}, N={n_seq})", fontsize=10)
            ax.set_aspect("equal")
            
            # Save output using strict coordinate filename stems
            filename = f"pca_class_means_L{seq_len}_N{n_seq}.pdf"
            save_figure(fig, PLOTS_DIR, filename)
            plt.close(fig)

    print("Grid search matrix sweep complete!")

if __name__ == "__main__":
    main()