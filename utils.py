"""Shared utilities for reproducing 'In-context learning of representations
can be explained by induction circuits.'"""

import os
import random
import functools

from matplotlib.colors import LogNorm
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import einops

from torch import Tensor
from jaxtyping import Float
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from model import TransformerConfig, Transformer
torch.set_grad_enabled(False)

# ── Constants ──────────────────────────────────────────────────────────────────

WORDS = [
    "apple", "bird", "car", "egg",
    "house", "milk", "plane", "opera",
    "box", "sand", "sun", "mango",
    "rock", "math", "code", "phone",
]

GRID_ROWS = 4
GRID_COLS = 4
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYER = 0
SEQ_LEN = 64
N_SEQUENCES = 16
SMOOTHING_WINDOW = 48

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
]

WORD_TO_COLOR = {word: color for word, color in zip(WORDS, COLORS)}


# ── Grid ───────────────────────────────────────────────────────────────────────

class Grid:
    def __init__(self, words=WORDS, rows=GRID_ROWS, cols=GRID_COLS):
        if rows * cols != len(words):
            raise ValueError(
                f"Grid dimensions ({rows}x{cols}={rows * cols}) "
                f"do not match number of words ({len(words)})"
            )
        self.words = words
        self.rows = rows
        self.cols = cols
        self.grid = np.array(words).reshape(rows, cols).tolist()
        self.word_to_row = {w: i // cols for i, w in enumerate(words)}
        self.word_to_col = {w: i % cols for i, w in enumerate(words)}

    # ── sequence generation ────────────────────────────────────────────────

    def generate_sequence(self, seq_len, start_word=None):
        """Random walk on the grid.  Optionally fix the starting word."""
        if start_word is not None:
            row, col = self.word_to_row[start_word], self.word_to_col[start_word]
        else:
            row, col = np.random.randint(0, self.rows), np.random.randint(0, self.cols)

        sequence = [self.grid[row][col]]
        while len(sequence) < seq_len:
            moves = self._valid_moves(row, col)
            direction = np.random.choice(moves)
            if direction == "up":    row -= 1
            elif direction == "down":  row += 1
            elif direction == "left":  col -= 1
            elif direction == "right": col += 1
            sequence.append(self.grid[row][col])
        return sequence

    # def generate_batch(self, seq_len):
    #     """16 sequences, each starting at a different grid word."""
    #     return [self.generate_sequence(seq_len, start_word=w) for w in self.words]

    def generate_batch(self, seq_len, n_sequences):
        """
        Generate a batch of random-walk sequences.

        Start words are assigned cyclically so that all grid words
        appear approximately equally often as starting points.
        """
        sequences = []

        for i in range(n_sequences):
            start_word = self.words[i % len(self.words)]
            sequences.append(
                self.generate_sequence(seq_len, start_word=start_word)
            )

        return sequences
    

    # ── adjacency ──────────────────────────────────────────────────────────

    def get_valid_next_words(self, word):
        row, col = self.word_to_row[word], self.word_to_col[word]
        next_words = []
        for move in self._valid_moves(row, col):
            if move == "up":    next_words.append(self.grid[row - 1][col])
            elif move == "down":  next_words.append(self.grid[row + 1][col])
            elif move == "left":  next_words.append(self.grid[row][col - 1])
            elif move == "right": next_words.append(self.grid[row][col + 1])
        return next_words

    def build_adjacency_matrix(self):
        """Return a 16x16 binary adjacency matrix (symmetric)."""
        n = len(self.words)
        A = np.zeros((n, n))
        for i, word in enumerate(self.words):
            for neighbor in self.get_valid_next_words(word):
                j = self.words.index(neighbor)
                A[i, j] = 1
        return A

    # ── internals ──────────────────────────────────────────────────────────

    def _valid_moves(self, row, col):
        moves = []
        if row > 0:              moves.append("up")
        if row < self.rows - 1:  moves.append("down")
        if col > 0:              moves.append("left")
        if col < self.cols - 1:  moves.append("right")
        return moves


# ── Seeding ────────────────────────────────────────────────────────────────────

def set_seed(seed, model_seed):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    if model_seed:
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(cache_dir=None, device=None):
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=device, cache_dir=cache_dir,
    )

from transformer_lens import HookedTransformer, HookedTransformerConfig

def load_toy_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define a small 1-layer, 4-head attention-only model
    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_head=32,
        n_ctx=SEQ_LEN,
        d_vocab=len(WORDS),
        act_fn=None,
        attention_dir="causal",
        attn_only=True,
        device=device,
        seed=42,
        normalization_type=None,
    )
    return HookedTransformer(cfg)


def load_JaydenModel(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TransformerConfig(
        d_vocab=len(WORDS),
        d_model=128,
        n_heads=1,
        d_head=128,
        n_layers=1,
        n_ctx=SEQ_LEN,
        d_mlp=512,
        attn_only=True,
        use_layernorm=False,
        use_bias=False,
        pos_embed="rope",
        use_causal_mask=True,
        model_seed=42,
    )
    return Transformer(cfg).to(device)

# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_sequence(model, sequence):
    """Tokenize a word sequence. Returns input_ids tensor."""
    text = " " + " ".join(sequence)
    return model.tokenizer(text, return_tensors="pt").input_ids


def simple_tokenize(sequence, vocab):
    # Map words to IDs, default to 0 if word is unknown
    tokens = [vocab.get(word, 0) for word in sequence.split()]
    return torch.tensor([tokens], dtype=torch.long)


# ── Accuracy ───────────────────────────────────────────────────────────────────

def get_model_accuracies(model, grid, sequence, fwd_hooks=[]):
    """Per-position probability assigned to valid next tokens."""
    tokens = tokenize_sequence(model, sequence)
    logits = model.run_with_hooks(tokens.to(model.cfg.device), fwd_hooks=fwd_hooks)
    probs = torch.softmax(logits, dim=-1)
    probs = probs[0, 1:, :]  # remove BOS position

    accuracies = []
    for i in range(len(sequence)):
        valid_next = grid.get_valid_next_words(sequence[i])
        token_ids = torch.tensor(
            [model.tokenizer.encode(" " + w, add_special_tokens=False) for w in valid_next]
        ).squeeze()
        accuracies.append(probs[i, token_ids].sum().item())
    return accuracies


# ── Activations ────────────────────────────────────────────────────────────────

def get_activations(model, sequence, layer, n_lookback, fwd_hooks=[]):
    """Return last n_lookback residual-stream activations at `layer`.

    Shape: [n_lookback, d_model]
    """
    tokens = tokenize_sequence(model, sequence)
    names_filter = [f"blocks.{layer}.hook_resid_pre"]

    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(
            tokens.to(model.cfg.device), names_filter=names_filter,
        )

    acts = cache[f"blocks.{layer}.hook_resid_pre"]  # [1, seq+1, d_model]
    # Remove BOS then take last n_lookback
    acts = acts[0, 1:, :]  # [seq_len, d_model]
    return acts[-n_lookback:, :]  # [n_lookback, d_model]


import torch
from transformer_lens import utils

def get_activations_toy_batch(model, sequences, layer, n_lookback, act_type="attn_out"):
    """
    Returns the last n_lookback activations for a BATCH of sequences.
    Output shape: [batch_size, n_lookback, d_model]
    """
    word_to_id = {word: i for i, word in enumerate(WORDS)}
    
    # Map all sequences in the batch to token IDs
    token_ids = [[word_to_id[word] for word in seq] for seq in sequences]
    tokens = torch.tensor(token_ids, dtype=torch.long).to(model.cfg.device)
    
    # Define the hook point
    from transformer_lens import utils
    hook_name = utils.get_act_name(act_type, layer)
    
    # Run with cache
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    acts = cache[hook_name] # Shape: [batch_size, seq_len, d_model]
    
    # Return the last n_lookback positions for ALL sequences in the batch
    # We drop the hardcoded acts[0] so we don't lose the batch dimension
    return acts[:, -n_lookback:, :]

def get_activations_Jayden_batch(model, sequences, layer, n_lookback):

    word_to_id = {word: i for i, word in enumerate(WORDS)}
    
    # Map all sequences in the batch to token IDs
    token_ids = [[word_to_id[word] for word in seq] for seq in sequences]
    tokens = torch.tensor(token_ids, dtype=torch.long)

    attn_output = []
    def hook_fn(module, input, output):
        attn_output.append(output[0])  

    hook_handle = model.blocks[0].attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(tokens)

    hook_handle.remove()
    sub_attn_out = attn_output[0]
    # Slice suffix
    sub_last_attn = sub_attn_out[:, -n_lookback:, :]

    return sub_last_attn

# ── PCA helpers ────────────────────────────────────────────────────────────────

def compute_class_means(
    activations: Float[Tensor, "n_lookback d_model"],
    sequence: list[str],
    words: list[str],
    n_lookback: int,
) -> Float[Tensor, "n_words d_model"]:
    """Mean activation per word over the last n_lookback positions."""
    tail = sequence[-n_lookback:]
    means = []
    for word in words:
        idxs = [i for i, w in enumerate(tail) if w == word]
        if idxs:
            means.append(activations[idxs].mean(dim=0))
        else:
            means.append(torch.zeros(activations.shape[-1], device=activations.device))
    return torch.stack(means)  # [16, d_model]


def compute_class_means_batch(activations_t, sequences, words, n_lookback):
    """
    Computes the mean activation vector for each word, averaging over all 
    occurrences across the lookback window AND the entire batch.
    
    activations_t: Tensor of shape [batch_size, n_lookback, d_model]
    sequences: List of lists of words, shape [batch_size, seq_len]
    """
    batch_size, n_look, d_model = activations_t.shape
    
    # Dictionary to collect activation vectors for each word
    word_vectors = {word: [] for word in words}
    
    # Loop over every sequence in the batch
    for b in range(batch_size):
        # Get the last n_lookback words for this specific sequence
        lookback_words = sequences[b][-n_look:] 
        
        # Match each lookback word with its corresponding activation vector
        for l in range(n_look):
            word = lookback_words[l]
            if word in word_vectors:
                word_vectors[word].append(activations_t[b, l, :])
                
    print(f"{word_vectors.keys()=}")
    # Compute the mean vector for each word across all batch occurrences
    class_means = []
    for word in words:
        vectors = word_vectors[word]
        if len(vectors) > 0:
            # Stack all collected occurrences and average them
            mean_vec = torch.stack(vectors).mean(dim=0)
        else:
            # Fallback tensor if a word never appeared in the lookback windows
            mean_vec = torch.zeros(d_model, device=activations_t.device)
        class_means.append(mean_vec)
        
    return torch.stack(class_means) # Shape: [len(words), d_model]


def compute_pca_directions(
    class_means: Float[Tensor, "n_words d_model"],
    top_n: int = 2,
) -> Float[Tensor, "top_n d_model"]:
    """PCA on the 16 class-mean vectors. Returns top_n right singular vectors."""
    centered = class_means - class_means.mean(dim=0, keepdim=True)
    _, _, V = torch.svd(centered)
    return einops.rearrange(V, "d_model n -> n d_model")[:top_n, :]


# ── Ablation hooks ─────────────────────────────────────────────────────────────

def head_ablation_hook(
    activation: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head: int,
    pos_mask: torch.Tensor = None,
) -> Float[Tensor, "batch seq n_heads d_head"]:
    if pos_mask is None:
        activation[:, :, head, :] = 0.0
    else:
        # pos_mask: [batch, seq] boolean tensor
        # Expand it to [batch, seq, d_head] to match activation[:, :, head, :]
        mask = pos_mask.unsqueeze(-1).expand(-1, -1, activation.shape[-1])
        activation[:, :, head, :][mask] = 0.0
    return activation


def make_ablation_hooks(heads_to_ablate, pos_mask=None):
    """Return a list of (name, hook_fn) pairs that zero-ablate the given heads.

    heads_to_ablate: list of (layer, head) tuples.
    pos_mask: boolean tensor of shape [batch, seq]. If True, ablates at that position.
    """
    return [
        (
            utils.get_act_name("z", layer),
            functools.partial(head_ablation_hook, head=head, pos_mask=pos_mask),
        )
        for layer, head in heads_to_ablate
    ]


# ── Plotting ───────────────────────────────────────────────────────────────────

def setup_plotting():
    plt.style.use("science")
    plt.rcParams.update({
        "text.usetex": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.top": False,
        "ytick.right": False,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.color": "gray",
        "grid.linewidth": 0.4,
        "savefig.bbox": "standard",  # override scienceplots' tight default
    })


def save_figure(fig, directory, filename):
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    fig.tight_layout()
    fig.savefig(os.path.join(directory, stem + ".pdf"))
    fig.savefig(os.path.join(directory, stem + ".png"), dpi=300)
    plt.close(fig)


def smooth(arr, window=SMOOTHING_WINDOW):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ── Plotly helpers ────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    autosize=False,
    margin=dict(l=60, r=40, t=60, b=60),
)

PLOTLY_AXIS = dict(
    showgrid=True, gridcolor="rgba(128,128,128,0.12)", gridwidth=0.4,
    zeroline=False, showline=True, linecolor="black", linewidth=1,
    ticks="outside", tickcolor="black",
)


def plotly_pca_layout(title):
    """Standard layout for PCA scatter plots in plotly."""
    return dict(
        title=title,
        width=600, height=600,
        xaxis=dict(title="PC1", scaleanchor="y", scaleratio=1, **PLOTLY_AXIS),
        yaxis=dict(title="PC2", **PLOTLY_AXIS),
        **PLOTLY_LAYOUT,
    )


def plotly_line_layout(title, xlabel, ylabel):
    """Standard layout for line plots in plotly."""
    return dict(
        title=title,
        width=700, height=400,
        xaxis=dict(title=xlabel, type="log", **PLOTLY_AXIS),
        yaxis=dict(title=ylabel, **PLOTLY_AXIS),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray", borderwidth=1),
        **PLOTLY_LAYOUT,
    )


def save_plotly(fig, directory, filename):
    """Save a plotly figure as HTML with fixed dimensions."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    fig.write_html(path, config={"responsive": False}, include_plotlyjs="cdn")
    print(f"Saved {path}")


def plotly_pca_traces(projected, grid, words=WORDS, word_to_color=WORD_TO_COLOR):
    """Return list of plotly traces for a PCA scatter: edges + star centroids."""
    import plotly.graph_objects as go
    traces = []
    A = grid.build_adjacency_matrix()
    # Collect all edges into a single trace with None breaks
    edge_x, edge_y = [], []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if A[i, j]:
                edge_x += [projected[i, 0].item(), projected[j, 0].item(), None]
                edge_y += [projected[i, 1].item(), projected[j, 1].item(), None]
    traces.append(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="gray", width=0.8, dash="dash"),
        opacity=0.4, showlegend=False, hoverinfo="skip",
    ))
    # Centroids
    for i, word in enumerate(words):
        traces.append(go.Scatter(
            x=[projected[i, 0].item()], y=[projected[i, 1].item()],
            mode="markers+text",
            marker=dict(size=16, symbol="star", color=word_to_color[word],
                        line=dict(width=1, color="black")),
            text=word, textposition="top right", textfont=dict(size=10),
            hovertemplate=f"centroid: <b>{word}</b><extra></extra>",
            hoverlabel=dict(bgcolor=word_to_color[word], font=dict(color="black")),
            showlegend=False,
        ))
    return traces

import math


def plot_attention_heatmaps(model, sequences, layer: int = 0, seq_index: int = 0,
                                cmap: str = 'viridis', vmin=None, vmax=None,
                                use_log_scale: bool = False, log_eps: float = 1e-12,
                                figsize=(20,16), save_path: str = None, show: bool = False):
    """
    Extracts and plots attention heatmaps for every head in a chosen layer 
    for our toy word-based transformer setup.
    
    sequences: List of lists of strings (e.g., [['The', 'cat', 'sat'], ...])
    """
    # 1. Convert string sequences to token IDs for the model
    word_to_id = {word: i for i, word in enumerate(WORDS)}
    token_ids = [[word_to_id[word] for word in seq] for seq in sequences]
    tokens = torch.tensor(token_ids, dtype=torch.long).to(model.cfg.device)
    
    # 2. Extract attention patterns from TransformerLens cache
    # hook_pattern shape is [batch, n_heads, query_pos, key_pos]
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    
    # Convert to numpy and select our specific sequence index
    layer_w = cache[hook_name].cpu().numpy()  # [batch, n_heads, T, T]
    
    batch, n_heads, T, _ = layer_w.shape
    print(f"{seq_index=}")
    if seq_index < 0 or seq_index >= batch:
        raise IndexError(f"seq_index out of range (0..{batch-1})")
        
    seq_w = layer_w[seq_index]  # [n_heads, T, T]
    
    # 3. Use the original string words as labels for the axes!
    labels = sequences[seq_index]

    # Grid layout calculations
    cols = min(4, n_heads) # Kept at 4 max for cleaner display of word labels
    rows = int(math.ceil(n_heads / cols))
    if figsize is None:
        figsize = (cols * 4.5, rows * 4.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    # Ensure axes is always a 2D array even if it's 1x1 or 1xN
    if n_heads == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np_atleast_2d_flat = np.atleast_2d(axes) 
    else:
        axes = np.atleast_2d(axes)

    # Log scaling calculations
    if use_log_scale:
        pos = seq_w[seq_w > 0]
        if pos.size > 0:
            auto_vmin = float(pos.min())
            auto_vmax = float(seq_w.max())
            vmin_used = (vmin if (vmin is not None and vmin > 0) else auto_vmin)
            vmax_used = (vmax if (vmax is not None and vmax > 0) else auto_vmax)
            norm = LogNorm(vmin=max(vmin_used, log_eps), vmax=max(vmax_used, log_eps))
        else:
            norm = None
    else:
        norm = None

    # Plot each attention head
    for h in range(n_heads):
        r = h // cols
        c = h % cols
        ax = axes[r, c]
        
        imshow_kwargs = dict(aspect='equal', cmap=cmap, interpolation='nearest')
        if norm is not None:
            imshow_kwargs['norm'] = norm
        else:
            if vmin is not None: imshow_kwargs['vmin'] = vmin
            if vmax is not None: imshow_kwargs['vmax'] = vmax

        im = ax.imshow(seq_w[h], **imshow_kwargs)
        ax.set_title(f"Layer {layer} | Head {h}", fontsize=11, fontweight='bold')
        
        # Apply the actual word string labels to the axes
        ax.set_xticks(np.arange(T))
        ax.set_yticks(np.arange(T))
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=9)
        
        ax.set_xlabel('Key Token (What it attends to)', fontsize=9)
        ax.set_ylabel('Query Token (What is looking)', fontsize=9)
            
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off any empty subplots in the grid
    for h in range(n_heads, rows * cols):
        r = h // cols
        c = h % cols
        axes[r, c].axis('off')

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig