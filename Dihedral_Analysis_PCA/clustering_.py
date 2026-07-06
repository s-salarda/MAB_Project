import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA

# ---------------------------
# SETTINGS
# ---------------------------
out_dir = "cluster_results"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------
# LOAD PCA DATA
# ---------------------------
print("Loading PCA data...")

with open("pca_output/Q_reduced_trimmed.data", "rb") as f:
    Q_reduced = pickle.load(f)

with open("pca_output/y_trimmed.data", "rb") as f:
    y = pickle.load(f)

# ---------------------------
# CLUSTERING (DBSCAN)
# ---------------------------
print("Running DBSCAN clustering...")

X = Q_reduced[:, :3]  # use PC1–PC3

db = DBSCAN(eps=0.4, min_samples=25)
clusters = db.fit_predict(X)

print("Clusters found:", np.unique(clusters))

# ---------------------------
# FIX NOISE COLOR (-1)
# ---------------------------
clusters_plot = clusters.copy()
if np.any(clusters_plot == -1):
    clusters_plot[clusters_plot == -1] = clusters_plot.max() + 1

# ---------------------------
# GENOTYPE LABELS
# ---------------------------
GENOTYPE_GROUP = {
    0: "WT", 1: "WT", 2: "WT",
    3: "D239N", 4: "D239N", 5: "D239N",
    6: "K637E", 7: "K637E", 8: "K637E"
}

print("Recomputing PCA object for variance...")

# Re-run PCA only to get variance (fast)
pca = PCA()
pca.fit(Q_reduced)

# =========================================================
# ✅ 2D PCA CLUSTER PLOTS
# =========================================================
def plot_pca_clusters_2d_discrete(Q_reduced, clusters, out_dir, pca,
                                 x_pc=0, y_pc=1, prefix="PCA_clusters"):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    os.makedirs(out_dir, exist_ok=True)

    # --- unique clusters ---
    uniq = np.unique(clusters)
    n = len(uniq)

    # --- discrete colormap ---
    base = plt.cm.get_cmap("tab20", n)
    cmap = ListedColormap([base(i) for i in range(n)])

    # map cluster labels -> 0..n-1
    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    c_idx = np.array([label_to_idx[c] for c in clusters])

    # ✅ --- DISCRETE NORMALIZATION (key part!)
    boundaries = np.arange(-0.5, n + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    # --- figure ---
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    x = Q_reduced[:, x_pc]
    y = Q_reduced[:, y_pc]

    # ✅ --- scatter (solid + outlined for density look)
    sc = ax.scatter(
        x,
        y,
        c=c_idx,
        cmap=cmap,
        norm=norm,
        edgecolors=[cmap(i) for i in c_idx],
        s=10,
        alpha=1
    )

    # ✅ --- tight zoom (matches 1nsRes)
    ax.set_xlim(x.min()*1.05, x.max()*1.05)
    ax.set_ylim(y.min()*1.05, y.max()*1.05)

    # ✅ --- variance labels (clean formatting)
    var_x = pca.explained_variance_ratio_[x_pc] * 100
    var_y = pca.explained_variance_ratio_[y_pc] * 100

    ax.set_xlabel(
        f"PC{x_pc+1} [ {var_x:.2f}% variance ]",
        fontsize=20, fontweight='bold'
    )

    ax.set_ylabel(
        f"PC{y_pc+1} [ {var_y:.2f}% variance ]",
        fontsize=20, fontweight='bold'
    )

    # ✅ --- title
    plt.title(
        f"PC{x_pc+1} vs PC{y_pc+1}",
        fontsize=20, fontweight='bold'
    )

    # ✅ --- ticks (thick + bold)
    ax.tick_params(width=2, length=7)
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

    # ✅ ✅ --- DISCRETE COLORBAR (fixed!)
    cbar = plt.colorbar(
        sc,
        ticks=np.arange(n),
        boundaries=boundaries,   # <-- THIS enforces block colors
        spacing='uniform'
    )

    cbar.ax.set_yticklabels([str(l) for l in uniq])
    cbar.set_label("Cluster ID", fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14, width=2)

    # --- save ---
    outfile = os.path.join(
        out_dir,
        f"{prefix}_PC{x_pc+1}_PC{y_pc+1}.png"
    )

    plt.savefig(outfile, dpi=400, bbox_inches="tight")
    plt.close()

    return outfile

# ---- Calls (make sure these are executed) ----
out_dir = "cluster_results"

print("Unique cluster labels:", np.unique(clusters))
print("Counts per label:", {c: int(np.sum(clusters == c)) for c in np.unique(clusters)})

plot_pca_clusters_2d_discrete(Q_reduced, clusters, out_dir, pca, 0, 1)
plot_pca_clusters_2d_discrete(Q_reduced, clusters, out_dir, pca, 0, 2)
plot_pca_clusters_2d_discrete(Q_reduced, clusters, out_dir, pca, 1, 2)

# =========================================================
# ✅ 3D PCA CLUSTER PLOT (INTERACTIVE)
# =========================================================
print("Generating 3D PCA plot...")

def plot_pca_clusters_3d_discrete(Q_reduced, clusters, out_dir, prefix="PCA_clusters_3D"):
    os.makedirs(out_dir, exist_ok=True)

    uniq = np.unique(clusters)
    n = len(uniq)

    base = plt.cm.get_cmap("tab20", n)
    palette = []
    for i in range(n):
        r, g, b, a = base(i)
        palette.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")

    lab_to_color = {lab: palette[i] for i, lab in enumerate(uniq)}
    point_colors = [lab_to_color[c] for c in clusters]

    fig = go.Figure()

    # main cloud
    fig.add_trace(go.Scatter3d(
        x=Q_reduced[:, 0],
        y=Q_reduced[:, 1],
        z=Q_reduced[:, 2],
        mode="markers",
        marker=dict(size=3, color=point_colors, opacity=0.85),
        text=[f"Cluster {c}" for c in clusters],
        hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
    ))

    # manual legend
    for lab in uniq:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=8, color=lab_to_color[lab]),
            name=f"Cluster {lab}"
        ))

    fig.update_layout(
        title="3D PCA",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        width=1000, height=800,
        legend=dict(itemsizing="constant")
    )

    outfile = os.path.join(out_dir, f"{prefix}.html")
    fig.write_html(outfile)
    return outfile

plot_pca_clusters_3d_discrete(Q_reduced, clusters, out_dir)

# =========================================================
# ✅ PIE CHARTS (GENOTYPE PER CLUSTER)
# =========================================================
print("Generating cluster pie charts...")

unique_clusters = sorted(set(clusters))
GENOTYPE_COLORS = {
    "WT": "#B9B9B9",
    "D239N": "#6dff83",
    "K637E": "#6f89ff"
}

for c in unique_clusters:
    if c == -1:
        continue

    idx = np.where(clusters == c)[0]
    genotypes = y[idx]

    group_labels = [GENOTYPE_GROUP[int(val)] for val in genotypes]
    counts = Counter(group_labels)

    labels = list(counts.keys())
    values = list(counts.values())

    colors = [GENOTYPE_COLORS[label] for label in labels]

    plt.figure(figsize=(6, 6))

    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        wedgeprops={'linewidth': 2, 'edgecolor': 'black'},
    )

    for autotext in autotexts:
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')
        autotext.set_color('black')

    for wedge in wedges:
        wedge.set_edgecolor('black')
        wedge.set_linewidth(2)


    for text in texts:
        text.set_fontsize(16)
        text.set_fontweight('bold')

    plt.title(
        f"Cluster {c}",
        fontsize=20,
        fontweight='bold'
    )

    outfile = os.path.join(out_dir, f"cluster_{c}_pie.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

print("✅ DONE.")

# -----------------------------
# PCA 2D no noise
# -----------------------------
def plot_pca_clusters_2d_discrete_no_noise(Q_reduced, clusters, out_dir, pca,
                                           x_pc=0, y_pc=1, prefix="PCA_clusters"):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    os.makedirs(out_dir, exist_ok=True)

    # --- SAME AS ORIGINAL ---
    uniq = np.unique(clusters)
    n = len(uniq)

    base = plt.cm.get_cmap("tab20", n)
    cmap = ListedColormap([base(i) for i in range(n)])

    label_to_idx = {lab: i for i, lab in enumerate(uniq)}

    # ------------------------------------------------------------
    # ✅ FILTER DATA (ONLY change!)
    # ------------------------------------------------------------
    mask = clusters != -1
    Qf = Q_reduced[mask]
    clusters_f = clusters[mask]

    # map colors using ORIGINAL mapping (important)
    c_idx_f = np.array([label_to_idx[c] for c in clusters_f])

    # --- SAME NORMALIZATION ---
    boundaries = np.arange(-0.5, n + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    # --- PLOT ---
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    x = Qf[:, x_pc]
    y = Qf[:, y_pc]

    sc = ax.scatter(
        x,
        y,
        c=c_idx_f,          # ✅ filtered colors
        cmap=cmap,
        norm=norm,
        edgecolors=[cmap(label_to_idx[c]) for c in clusters_f],
        s=10,
        alpha=1
    )

    # ✅ SAME AXES STYLE
    ax.set_xlim(x.min()*1.05, x.max()*1.05)
    ax.set_ylim(y.min()*1.05, y.max()*1.05)

    var_x = pca.explained_variance_ratio_[x_pc] * 100
    var_y = pca.explained_variance_ratio_[y_pc] * 100

    ax.set_xlabel(
        f"PC{x_pc+1} [ {var_x:.2f}% variance ]",
        fontsize=20, fontweight='bold'
    )
    ax.set_ylabel(
        f"PC{y_pc+1} [ {var_y:.2f}% variance ]",
        fontsize=20, fontweight='bold'
    )

    plt.title(f"PC{x_pc+1} vs PC{y_pc+1}",
              fontsize=20, fontweight='bold')

    ax.tick_params(width=2, length=7)
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

    # ✅ COLORBAR UNCHANGED (still shows -1)
    cbar = plt.colorbar(
        sc,
        ticks=np.arange(n),
        boundaries=boundaries,
        spacing='uniform'
    )

    cbar.ax.set_yticklabels([str(l) for l in uniq])  # includes -1
    cbar.set_label("Cluster ID", fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14, width=2)

    outfile = os.path.join(
        out_dir,
        f"{prefix}_PC{x_pc+1}_PC{y_pc+1}_noNoise.png"
    )

    plt.savefig(outfile, dpi=400, bbox_inches="tight")
    plt.close()

    return outfile

plot_pca_clusters_2d_discrete_no_noise(Q_reduced, clusters, out_dir, pca, 0, 1)
plot_pca_clusters_2d_discrete_no_noise(Q_reduced, clusters, out_dir, pca, 0, 2)
plot_pca_clusters_2d_discrete_no_noise(Q_reduced, clusters, out_dir, pca, 1, 2)

# ---------------------
# PCA 3D no Noise 
# ---------------------
def plot_pca_clusters_3d_discrete_no_noise_points(Q_reduced, clusters, out_dir, prefix="PCA_clusters_3D"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    os.makedirs(out_dir, exist_ok=True)

    # ✅ Keep ORIGINAL legend/color mapping (includes -1 if present)
    uniq = np.unique(clusters)
    n = len(uniq)

    base = plt.cm.get_cmap("tab20", n)
    palette = []
    for i in range(n):
        r, g, b, a = base(i)
        palette.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")

    lab_to_color = {lab: palette[i] for i, lab in enumerate(uniq)}

    # ------------------------------------------------------------
    # ✅ ONLY CHANGE: filter OUT noise points from the main scatter
    # ------------------------------------------------------------
    mask = clusters != -1
    Qf = Q_reduced[mask]
    clusters_f = clusters[mask]

    # point colors using ORIGINAL mapping (so colors don't shift)
    point_colors = [lab_to_color[c] for c in clusters_f]

    fig = go.Figure()

    # ✅ main cloud (NO NOISE POINTS)
    fig.add_trace(go.Scatter3d(
        x=Qf[:, 0],
        y=Qf[:, 1],
        z=Qf[:, 2],
        mode="markers",
        marker=dict(size=3, color=point_colors, opacity=0.85),
        text=[f"Cluster {c}" for c in clusters_f],
        hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
    ))

    # ✅ manual legend UNCHANGED (still shows -1, 0, 1, ...)
    for lab in uniq:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=8, color=lab_to_color[lab]),
            name=f"Cluster {lab}"
        ))

    fig.update_layout(
        title="3D PCA",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        width=1000, height=800,
        legend=dict(itemsizing="constant")
    )

    outfile = os.path.join(out_dir, f"{prefix}_noNoisePoints.html")
    fig.write_html(outfile)
    return outfile

plot_pca_clusters_3d_discrete_no_noise_points(Q_reduced, clusters, out_dir)