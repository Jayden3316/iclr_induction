"""Figure 5: toy model of previous-token (neighbor) mixing."""

import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import einops

from utils import (
    WORDS, WORD_TO_COLOR,
    Grid, set_seed, setup_plotting, save_figure,
    save_plotly,
)

DATA_DIR = "results/neighbor_mixing/data"
PLOTS_DIR = "results/neighbor_mixing/plots"
D_EMBED = 4096


def pca_nd(embeddings, dim=3):
    """Mean-center and SVD → project onto top-n PCs. Returns [16, dim]."""
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, V = torch.svd(centered)
    directions = einops.rearrange(V, "d n -> n d")[:dim, :]  # [dim, d]
    return centered @ directions.T  # [16, dim]

def plot_pca_scatter(projected, grid, title, filename):
    """Scatter of 16 word embeddings projected onto PCs with grid edges."""
    dim = projected.shape[1]
    is_3d = (dim == 3)
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                if is_3d:
                    ax.plot(
                        [projected[i, 0].item(), projected[j, 0].item()],
                        [projected[i, 1].item(), projected[j, 1].item()],
                        [projected[i, 2].item(), projected[j, 2].item()],
                        color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                    )
                else:
                    ax.plot(
                        [projected[i, 0].item(), projected[j, 0].item()],
                        [projected[i, 1].item(), projected[j, 1].item()],
                        color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                    )

    for i, word in enumerate(WORDS):
        if is_3d:
            ax.scatter(
                projected[i, 0].item(), projected[i, 1].item(), projected[i, 2].item(),
                color=WORD_TO_COLOR[word], s=120, marker="*",
                edgecolors="black", linewidths=0.5, zorder=5,
            )
            ax.text(
                projected[i, 0].item(), projected[i, 1].item(), projected[i, 2].item(),
                word, size=8, zorder=1, color='k'
            )
        else:
            ax.scatter(
                projected[i, 0].item(), projected[i, 1].item(),
                color=WORD_TO_COLOR[word], s=120, marker="*",
                edgecolors="black", linewidths=0.5, zorder=5,
            )
            ax.text(
                projected[i, 0].item(), projected[i, 1].item(),
                word, size=8, zorder=1, color='k'
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if is_3d:
        ax.set_zlabel("PC3")
    ax.set_title(title)
    
    save_figure(fig, PLOTS_DIR, filename)
    print(f"Saved {filename}")

def plot_pca_evolution_plotly(projections, grid, title, filename):
    """Animated Plotly scatter of 16 word embeddings over multiple iterations."""
    dim = projections.shape[2]
    is_3d = (dim == 3)
    A = grid.build_adjacency_matrix()
    num_steps = projections.shape[0]

    def get_traces(step_idx):
        proj = projections[step_idx]
        traces = []
        edge_x, edge_y, edge_z = [], [], []
        for i in range(len(WORDS)):
            for j in range(i + 1, len(WORDS)):
                if A[i, j]:
                    edge_x += [proj[i, 0].item(), proj[j, 0].item(), None]
                    edge_y += [proj[i, 1].item(), proj[j, 1].item(), None]
                    if is_3d:
                        edge_z += [proj[i, 2].item(), proj[j, 2].item(), None]
        
        if is_3d:
            traces.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode="lines",
                line=dict(color="gray", width=2, dash="dash"),
                opacity=0.4, showlegend=False, hoverinfo="skip",
            ))
        else:
            traces.append(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(color="gray", width=2, dash="dash"),
                opacity=0.4, showlegend=False, hoverinfo="skip",
            ))
        
        for i, word in enumerate(WORDS):
            if is_3d:
                traces.append(go.Scatter3d(
                    x=[proj[i, 0].item()], y=[proj[i, 1].item()], z=[proj[i, 2].item()],
                    mode="markers+text",
                    marker=dict(size=6, symbol="circle", color=WORD_TO_COLOR[word],
                                line=dict(width=1, color="black")),
                    text=word, textposition="top right", textfont=dict(size=10),
                    hovertemplate=f"centroid: <b>{word}</b><extra></extra>",
                    showlegend=False,
                ))
            else:
                traces.append(go.Scatter(
                    x=[proj[i, 0].item()], y=[proj[i, 1].item()],
                    mode="markers+text",
                    marker=dict(size=6, symbol="circle", color=WORD_TO_COLOR[word],
                                line=dict(width=1, color="black")),
                    text=word, textposition="top right", textfont=dict(size=10),
                    hovertemplate=f"centroid: <b>{word}</b><extra></extra>",
                    showlegend=False,
                ))
        return traces

    fig = go.Figure(data=get_traces(0))

    frames = []
    for k in range(num_steps):
        frames.append(go.Frame(data=get_traces(k), name=str(k)))
    fig.frames = frames

    sliders = [{
        "steps": [
            {
                "method": "animate",
                "args": [
                    [str(k)],
                    {"mode": "immediate", "frame": {"duration": 500, "redraw": True}, "transition": {"duration": 300}}
                ],
                "label": f"{k}"
            }
            for k in range(num_steps)
        ],
        "transition": {"duration": 300},
        "currentvalue": {"font": {"size": 12}, "prefix": "Iteration: ", "visible": True, "xanchor": "right"},
        "len": 0.9,
        "x": 0.1,
        "pad": {"b": 10, "t": 50}
    }]

    layout_args = dict(
        title=title,
        width=800, height=800,
        margin=dict(l=0, r=0, b=80, t=40),
        sliders=sliders,
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "showactive": False,
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
            "pad": {"t": 50, "r": 10},
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                }
            ]
        }]
    )
    
    if is_3d:
        layout_args["scene"] = dict(
            xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
            xaxis=dict(range=[projections[:,:,0].min()-1, projections[:,:,0].max()+1]),
            yaxis=dict(range=[projections[:,:,1].min()-1, projections[:,:,1].max()+1]),
            zaxis=dict(range=[projections[:,:,2].min()-1, projections[:,:,2].max()+1]),
        )
    else:
        layout_args["xaxis_title"] = 'PC1'
        layout_args["yaxis_title"] = 'PC2'
        layout_args["xaxis"] = dict(range=[projections[:,:,0].min()-1, projections[:,:,0].max()+1])
        layout_args["yaxis"] = dict(range=[projections[:,:,1].min()-1, projections[:,:,1].max()+1])
        
    fig.update_layout(**layout_args)
    
    html_stem = os.path.splitext(filename)[0]
    save_plotly(fig, PLOTS_DIR, f"{html_stem}.html")


def main():
    parser = argparse.ArgumentParser(description="Toy model of previous-token mixing")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3, help="PCA dimension (2 or 3)")
    parser.add_argument("--ablate", type=str, default=None, help="Word to ablate (or 'all')")
    args = parser.parse_args()

    setup_plotting()
    grid = Grid()

    dim = args.dim
    words_to_ablate = [None]
    if args.ablate == "all":
        words_to_ablate = WORDS
    elif args.ablate is not None:
        if args.ablate not in WORDS:
            raise ValueError(f"Unknown word to ablate: {args.ablate}")
        words_to_ablate = [args.ablate]

    for ablated_word in words_to_ablate:
        ablate_idx = None
        if ablated_word is not None:
            ablate_idx = WORDS.index(ablated_word)

        suffix = f"_{dim}d"
        if ablated_word is not None:
            suffix += f"_ablated_{ablated_word}"

        data_path = os.path.join(DATA_DIR, f"mixing{suffix}_loop.npz")
        num_iterations = 10

        if os.path.exists(data_path):
            print(f"Loading cached data (delete data/ to recompute)...")
            data = np.load(data_path)
            projections = data["projections"]
        else:
            A = grid.build_adjacency_matrix()
            A_torch = torch.tensor(A, dtype=torch.float32)

            # Random Gaussian embeddings
            set_seed(42)
            embeddings = torch.randn(16, D_EMBED)

            projections = [pca_nd(embeddings, dim=dim).numpy()]
            current_embeddings = embeddings
            degree = A_torch.sum(dim=1, keepdim=True)  # [16, 1]

            for _ in range(num_iterations):
                neighbor_sum = A_torch @ current_embeddings
                mixed = current_embeddings + neighbor_sum / degree
                
                if ablate_idx is not None:
                    mixed[ablate_idx] = current_embeddings[ablate_idx]
                    
                projections.append(pca_nd(mixed, dim=dim).numpy())
                current_embeddings = mixed

            projections = np.stack(projections) # [11, 16, dim]

            os.makedirs(DATA_DIR, exist_ok=True)
            np.savez(data_path, projections=projections)
            print(f"Cached {data_path}")

        # ── Plotting ──────────────────────────────────────────────────────────────
        title_suffix = "" if ablated_word is None else f"\n(Ablated: {ablated_word})"
        
        plot_pca_scatter(projections[0], grid,
                         f"Random embeddings\n(no neighbor mixing){title_suffix}", f"before_mixing{suffix}.pdf")
        plot_pca_scatter(projections[-1], grid,
                         f"Random embeddings\n(after {num_iterations} rounds of neighbor mixing){title_suffix}", f"after_mixing{suffix}.pdf")
        
        # Animated plotly visualization
        plot_pca_evolution_plotly(projections, grid,
                         f"Neighbor Mixing Evolution{title_suffix}", f"mixing_evolution{suffix}.html")


if __name__ == "__main__":
    main()
