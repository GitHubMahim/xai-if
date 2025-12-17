import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'tool-data')
VIS_PNG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization-tool-pngs')


def load_influences(path: str) -> np.ndarray:
    """Load influences over time.

    Expects an array shaped (num_timestamps, num_train).
    """
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        # Some saves might wrap a list; convert to array
        arr = np.stack(arr, axis=0)
    assert arr.ndim == 2, f"Expected 2D array (T, N_train); got shape {arr.shape}"
    return arr


def zscore_time_series(series_2d: np.ndarray) -> np.ndarray:
    """Z-score each training-point series across time (column-wise).

    series_2d: shape (T, N)
    returns standardized: shape (T, N)
    """
    # Column-wise mean/std across time axis 0
    mu = series_2d.mean(axis=0, keepdims=True)
    sigma = series_2d.std(axis=0, keepdims=True) + 1e-8
    return (series_2d - mu) / sigma


def cluster_with_tsne_kmeans(series_2d: np.ndarray,
                              n_components: int = 2,
                              n_clusters: int = 20,
                              random_state: int = 0,
                              perplexity: int = 30):
    """Cluster time series using t-SNE (on z-scored data) + k-means.

    Returns:
      labels: (N,) cluster label per training point
      tsne: fitted TSNE instance
      km: fitted KMeans instance
      X_tsne: (N, n_components) embedding used by k-means
    """
    # z-score per series (cluster by shape, not magnitude)
    Z = zscore_time_series(series_2d)
    # Reshape to (N, T) for t-SNE/sklearn
    Z_NT = Z.T  # (N, T)

    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    X_tsne = tsne.fit_transform(Z_NT)  # (N, n_components)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_tsne)
    return labels, tsne, km, X_tsne


def compute_cluster_means(raw_series_2d: np.ndarray, labels: np.ndarray):
    """Compute mean time series per cluster in RAW (unstandardized) space.

    raw_series_2d: (T, N)
    returns dict: {cluster_id: mean_series (T,)} and sizes
    """
    T, N = raw_series_2d.shape
    means = {}
    sizes = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        sizes[int(c)] = int(idx.size)
        if idx.size == 0:
            means[int(c)] = np.zeros(T)
        else:
            means[int(c)] = raw_series_2d[:, idx].mean(axis=1)
    return means, sizes


def plot_time_clusters(influences_over_time: np.ndarray,
                       labels: np.ndarray,
                       means: dict,
                       sizes: dict,
                       sample_per_cluster: int = 0,
                       title: str = "Influence over Time (Clustered)"):
    """Plot cluster mean lines (optionally a few samples per cluster).

    influences_over_time: (T, N)
    """
    T, N = influences_over_time.shape
    xs = np.arange(T)

    # Color cycle per cluster
    unique = sorted(means.keys())
    cmap = plt.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, c in enumerate(unique):
        color = cmap(i % 20)
        ax.plot(xs, means[c], color=color, linewidth=2.2, label=f"Cluster {c} (n={sizes[c]})")

        if sample_per_cluster > 0:
            idx = np.where(labels == c)[0]
            if idx.size > 0:
                sel = idx[: min(sample_per_cluster, idx.size)]
                ax.plot(xs, influences_over_time[:, sel], color=color, alpha=0.12, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Influence value")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    # Add interactive zoom on X with auto Y scaling using SpanSelector
    lines = list(ax.lines)  # capture the Line2D artists

    def onselect(xmin, xmax):
        # If selection is very small, ignore
        if xmin is None or xmax is None or abs(xmax - xmin) < 1e-6:
            return
        # Clamp to data range
        xmin = max(xmin, xs.min())
        xmax = min(xmax, xs.max())
        if xmax <= xmin:
            return
        mask = (xs >= xmin) & (xs <= xmax)
        if not np.any(mask):
            return
        ymin, ymax = np.inf, -np.inf
        for ln in lines:
            y = ln.get_ydata()
            # Handle lines that may be shorter/longer
            L = min(len(y), len(xs))
            y_slice = y[:L][mask[:L]]
            if y_slice.size:
                ymin = min(ymin, np.nanmin(y_slice))
                ymax = max(ymax, np.nanmax(y_slice))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return
        # Add a small margin
        span = ymax - ymin
        pad = 0.05 * (span if span > 0 else 1.0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin - pad, ymax + pad)
        fig.canvas.draw_idle()

    span = SpanSelector(ax, onselect, direction='horizontal', useblit=True,
                        props=dict(alpha=0.2, facecolor='tab:blue'))

    # Reset view button
    reset_ax = plt.axes([0.88, 0.92, 0.1, 0.05])
    reset_btn = Button(reset_ax, 'Reset', hovercolor='0.85')

    def on_reset(event):
        ax.set_xlim(xs.min(), xs.max())
        # Recompute y-lims on full range (means only for robustness)
        y_all = np.array([means[c] for c in unique])
        ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
        span = ymax - ymin
        pad = 0.05 * (span if span > 0 else 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
        fig.canvas.draw_idle()

    reset_btn.on_clicked(on_reset)

    # Instruction
    ax.text(0.01, 0.99, 'Drag to zoom X; Reset to restore', transform=ax.transAxes,
            va='top', ha='left', fontsize=9, color='0.3')

    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.9])
    
    # Save the plot before showing
    os.makedirs(VIS_PNG_DIR, exist_ok=True)
    output_filename = os.path.join(VIS_PNG_DIR, 'influence_time_line_graph.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_filename}'")
    
    plt.show()


def main():
    # Config
    input_path = os.path.join(TOOL_DATA_DIR, 'index_15533_influences.npy')   # (T, N_train)
    n_components = 2
    n_clusters = 5
    sample_per_cluster = 0  # set to >0 to draw thin sample lines
    perplexity = 30  # t-SNE perplexity parameter

    print(f"Loading influences from '{input_path}'...")
    influ_tn = load_influences(input_path)  # (T, N)
    T, N = influ_tn.shape
    print(f"Loaded influences shape: {influ_tn.shape} (T={T}, N_train={N})")

    print("Clustering with t-SNE + KMeans...")
    labels, tsne, km, X_tsne = cluster_with_tsne_kmeans(
        influ_tn, n_components=n_components, n_clusters=n_clusters, 
        random_state=0, perplexity=perplexity
    )
    means, sizes = compute_cluster_means(influ_tn, labels)

    print("Plotting clusters...")
    plot_time_clusters(
        influences_over_time=influ_tn,
        labels=labels,
        means=means,
        sizes=sizes,
        sample_per_cluster=sample_per_cluster,
        title=f"Influence over Time - t-SNE({n_components}D) + KMeans(k={n_clusters})"
    )


if __name__ == "__main__":
    main()


