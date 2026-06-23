"""Extended ablation experiments on induction and previous-token heads."""

import os
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import einops
import tqdm

from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR,
    Grid, set_seed, load_model, tokenize_sequence,
    get_model_accuracies, get_activations, compute_class_means,
    compute_pca_directions, make_ablation_hooks, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)

DATA_DIR = "results/ablation_extended/data"
PLOTS_DIR = "results/ablation_extended/plots"
N_LOOKBACK = 200
N_TEST_SEQS = 32
REPEAT_LEN = 32

def compute_head_scores(model):
    """Score every head for induction and previous-token behaviour."""
    set_seed(42)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    induction_scores = np.zeros((n_layers, n_heads))
    prev_token_scores = np.zeros((n_layers, n_heads))

    name_filters = [f"blocks.{l}.attn.hook_pattern" for l in range(n_layers)]

    for seq_idx in tqdm.tqdm(range(N_TEST_SEQS), desc="Identifying heads"):
        random_tokens = torch.randint(0, model.cfg.d_vocab // 2, (REPEAT_LEN,))
        repeated = einops.repeat(random_tokens, "s -> (two s)", two=2)

        _, cache = model.run_with_cache(repeated, names_filter=name_filters)

        for layer in range(n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]
            for head in range(n_heads):
                p = pattern[head]
                offset = REPEAT_LEN - 1
                diag = p.diagonal(offset=-offset)
                induction_scores[layer, head] += diag[offset:].mean().item()

                diag_prev = p.diagonal(offset=-1)
                prev_token_scores[layer, head] += diag_prev[1:].mean().item()
        del cache

    induction_scores /= N_TEST_SEQS
    prev_token_scores /= N_TEST_SEQS
    return induction_scores, prev_token_scores

def scores_to_ranked(scores):
    n_layers, n_heads = scores.shape
    return sorted(
        [(scores[l, h], l, h) for l in range(n_layers) for h in range(n_heads)],
        reverse=True,
    )

def print_top_heads(induction_ranked, prev_token_ranked, k=32):
    print(f"\nTop-{k} induction heads:")
    for score, layer, head in induction_ranked[:k]:
        print(f"  L{layer:>2d}.H{head:>2d}  score={score:.4f}")
    print(f"\nTop-{k} previous-token heads:")
    for score, layer, head in prev_token_ranked[:k]:
        print(f"  L{layer:>2d}.H{head:>2d}  score={score:.4f}")


def run_accuracy_sweep(model, grid, heads_to_ablate, label, target_token=None):
    """Run ablation accuracy for a specific list of heads, optionally targeted at a token."""
    set_seed(42)
    sequences = grid.generate_batch(SEQ_LEN)
    all_accs = []
    
    for seq in tqdm.tqdm(sequences, desc=f"Ablating {label}"):
        pos_mask = None
        if target_token is not None:
            # Create a boolean mask where True means the token at this position is the target token
            # Sequence has length SEQ_LEN. Tokenizer adds BOS token.
            # So tokens has shape [1, SEQ_LEN + 1]. The mask needs to match the sequence length.
            # We want to ablate when the current token is `target_token`.
            tokens = tokenize_sequence(model, seq)
            # The tokens tensor is shape [1, seq_len + 1].
            # We need to find positions corresponding to target_token.
            # To be exact, we can use the original string sequence:
            # Note: the input seq is a list of strings, e.g., ["bird", "apple", ...].
            # len(seq) == SEQ_LEN.
            # The tokenized sequence has a BOS token, so its length is SEQ_LEN + 1.
            # activation in the hook has shape [batch, seq_len + 1, n_heads, d_head]
            mask_list = [False] # False for BOS
            for word in seq:
                mask_list.append(word == target_token)
            pos_mask = torch.tensor([mask_list], dtype=torch.bool, device=model.cfg.device)
            
        hooks = make_ablation_hooks(heads_to_ablate, pos_mask=pos_mask)
        all_accs.append(get_model_accuracies(model, grid, seq, fwd_hooks=hooks))
        
    return np.array(all_accs).mean(axis=0)


def compute_pca_projected(model, grid, sequence, heads_to_ablate, target_token=None):
    pos_mask = None
    if target_token is not None:
        mask_list = [False]
        for word in sequence:
            mask_list.append(word == target_token)
        pos_mask = torch.tensor([mask_list], dtype=torch.bool, device=model.cfg.device)
        
    hooks = make_ablation_hooks(heads_to_ablate, pos_mask=pos_mask) if heads_to_ablate else []
    activations = get_activations(model, sequence, LAYER, N_LOOKBACK, fwd_hooks=hooks)
    class_means = compute_class_means(activations, sequence, WORDS, N_LOOKBACK)
    pca_dirs = compute_pca_directions(class_means, top_n=2)
    projected = (class_means @ pca_dirs.T).cpu().numpy()
    return projected


def main():
    parser = argparse.ArgumentParser(description="Extended Ablation Experiments")
    parser.add_argument("--heads", type=str, default=None, 
                        help="Comma-separated list of layer,head pairs to ablate (e.g. '24,5,25,2'). If provided, overrides top-k.")
    parser.add_argument("--top_k", type=int, default=16, 
                        help="Number of top heads to ablate if --heads is not specified.")
    parser.add_argument("--target_token", type=str, default=None, 
                        help="If provided (e.g. 'bird'), only ablate when the current token is this word.")
    parser.add_argument("--head_type", type=str, choices=["induction", "prev_token"], default="induction",
                        help="Which type of top-k heads to ablate if --heads is not specified.")
    args = parser.parse_args()

    setup_plotting()
    grid = Grid()
    os.makedirs(DATA_DIR, exist_ok=True)
    model = load_model()

    # Step 1: Head identification
    head_path = os.path.join(DATA_DIR, "head_scores.npz")
    if os.path.exists(head_path):
        print("Loading cached head scores...")
        data = np.load(head_path)
        induction_scores, prev_token_scores = data["induction_scores"], data["prev_token_scores"]
    else:
        induction_scores, prev_token_scores = compute_head_scores(model)
        np.savez(head_path, induction_scores=induction_scores, prev_token_scores=prev_token_scores)
        print(f"Cached {head_path}")

    induction_ranked = scores_to_ranked(induction_scores)
    prev_token_ranked = scores_to_ranked(prev_token_scores)

    heads_to_ablate = []
    label = ""
    if args.heads:
        parts = args.heads.split(",")
        if len(parts) % 2 != 0:
            raise ValueError("Heads must be provided as layer,head pairs")
        for i in range(0, len(parts), 2):
            heads_to_ablate.append((int(parts[i]), int(parts[i+1])))
        label = f"Explicit heads: {args.heads}"
    else:
        if args.head_type == "induction":
            heads_to_ablate = [(l, h) for _, l, h in induction_ranked[:args.top_k]]
            label = f"Top-{args.top_k} induction heads"
        else:
            heads_to_ablate = [(l, h) for _, l, h in prev_token_ranked[:args.top_k]]
            label = f"Top-{args.top_k} prev-token heads"

    print(f"\nAblating {label}")
    if args.target_token:
        print(f"Target token condition: '{args.target_token}'")

    # Baseline (no ablation)
    baseline_accs = run_accuracy_sweep(model, grid, [], "Baseline")
    
    # Ablated
    ablated_accs = run_accuracy_sweep(model, grid, heads_to_ablate, label, target_token=args.target_token)

    # Plot Accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(smooth(baseline_accs), color="black", label="No ablation")
    
    abl_label = label + (f" (only on '{args.target_token}')" if args.target_token else "")
    ax.plot(smooth(ablated_accs), color="red", label=abl_label)
    
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Accuracy")
    ax.legend(fontsize=8)
    
    plot_name = "ablation_extended_acc.pdf"
    if args.target_token:
        plot_name = f"ablation_extended_acc_{args.target_token}.pdf"
    save_figure(fig, PLOTS_DIR, plot_name)
    print(f"Saved accuracy plot to {plot_name}")

    # PCA
    seq_path = os.path.join(DATA_DIR, "sequence_ext.json")
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            sequence = json.load(f)
    else:
        set_seed(42)
        sequence = grid.generate_sequence(SEQ_LEN)
        with open(seq_path, "w") as f:
            json.dump(sequence, f)

    projected_baseline = compute_pca_projected(model, grid, sequence, [])
    projected_ablated = compute_pca_projected(model, grid, sequence, heads_to_ablate, target_token=args.target_token)

    def _plot_pca(projected, title, filename):
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
            ax.scatter(projected[i, 0].item(), projected[i, 1].item(),
                       color=WORD_TO_COLOR[word], s=120, marker="*",
                       edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(word, (projected[i, 0].item(), projected[i, 1].item()),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        save_figure(fig, PLOTS_DIR, filename)

    pca_base_name = "pca_extended_baseline.pdf"
    pca_abl_name = "pca_extended_ablated.pdf"
    if args.target_token:
        pca_abl_name = f"pca_extended_ablated_{args.target_token}.pdf"

    _plot_pca(projected_baseline, "No ablation", pca_base_name)
    print(f"Saved PCA baseline plot to {pca_base_name}")
    _plot_pca(projected_ablated, abl_label, pca_abl_name)
    print(f"Saved PCA ablated plot to {pca_abl_name}")

if __name__ == "__main__":
    main()
