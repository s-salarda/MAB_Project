import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from matplotlib.lines import Line2D

# ================================================================
# GLOBAL PLOTTING STYLE
# ================================================================

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.fontsize": 13,
    "figure.dpi": 200,
})

# ================================================================
# FILE LOADING
# ================================================================

def load_structure_files(folder_path, keyword=None):
    """
    Load all *.sum.dat files from a folder and return them as DataFrames.

    """
    dat_files = [f for f in os.listdir(folder_path) if f.endswith("sum.dat")]
    if keyword:
        dat_files = [f for f in dat_files if keyword.lower() in f.lower()]

    dfs = {}
    for file in dat_files:
        df = pd.read_csv(os.path.join(folder_path, file), sep=r"\s+", header=None)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        dfs[file] = df
    return dfs

# ================================================================
# STRUCTURE-LEVEL SUMMARY
# ================================================================

def entire_simulation_percentages(folder_path, keyword=None, plot=True):
    """
    Compute average secondary structure percentages across all frames
    for all replicates of a genotype.

    """
    dfs = load_structure_files(folder_path, keyword)

    structure_cols = ["Extended", "Bridge", "3-10", "Alpha", "Pi", "Turn", "Bend"]
    desired_order = ["None", "Bridge", "Extended", "3-10", "Alpha", "Pi", "Turn", "Bend"]

    avg_results = []

    for _, df in dfs.items():
        df_numeric = df.apply(pd.to_numeric, errors="coerce")
        df_struct = df_numeric[structure_cols]

        col_means = df_struct.mean() * 100
        none_value = max(0.0, 100.0 - col_means.sum())
        col_means["None"] = none_value

        ordered = col_means.reindex(desired_order)
        avg_results.append(ordered)

    combined = pd.concat(avg_results, axis=1).T
    avg_across_sims = combined.mean()

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        bars = ax.bar(avg_across_sims.index, avg_across_sims.values)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center")

        ax.set_title("Secondary Structure Summary", pad=30)
        ax.set_ylabel("Percent of Simulation (%)", labelpad=18)
        ax.set_xlabel("Structure Type", labelpad=18)
        ax.set_xticklabels(avg_across_sims.index, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout(pad=2.0)
        plt.show()

    return avg_across_sims

# ================================================================
# STRUCTURE-LEVEL GROUPED BAR PLOTS
# ================================================================

def plot_loop2_organization(genotype_data, title="Loop 2 Organization"):
    """
    Create a grouped bar plot comparing secondary structure percentages
    across WT, D239N, and K637E for all structure types.

    """
    structure_keys = ["None", "Bridge", "Extended", "3-10", "Alpha", "Pi", "Turn", "Bend"]
    structure_labels = ["None", "Parallel\nBeta-Sheet", "Anti-parallel\nBeta-sheet",
                        "3-10 Helix", "Alpha helix", "Pi (3-14) helix", "Turn", "Bend"]

    genotypes = list(genotype_data.keys())
    color_map = {"WT": "#343436", "D239N": "#ff0101", "K637E": "#0e8400"}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    bar_width = 0.025
    bar_spacing = 0.015
    group_spacing = 0.15

    num_structures = len(structure_keys)
    num_genotypes = len(genotypes)
    group_width = num_genotypes * bar_width + (num_genotypes - 1) * bar_spacing
    x = np.arange(num_structures) * (group_width + group_spacing)

    for i, genotype in enumerate(genotypes):
        values = [genotype_data[genotype].get(struct, 0.0) for struct in structure_keys]
        positions = x + i * (bar_width + bar_spacing)

        ax.bar(positions, values, width=bar_width,
               color=color_map.get(genotype, "#666666"),
               edgecolor="black", linewidth=1.8, alpha=0.7, label=genotype)

        for pos, val in zip(positions, values):
            ax.text(pos, val + 1, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, rotation=90)

    ax.set_xticks(x + (group_width - bar_width) / 2)
    ax.set_xticklabels(structure_labels, rotation=45, ha="right")

    ax.set_ylabel("Percent of Simulations", labelpad=18)
    ax.set_xlabel("Secondary Structure Type", labelpad=18)
    ax.set_title(title, pad=32)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(pad=2.0)
    plt.show()

def plot_loop2_organization2(genotype_data, title="Loop 2 Beta-Sheet Organization"):
    """
    Create a grouped bar plot comparing only beta-sheet structures
    (parallel vs anti-parallel) across WT, D239N, and K637E.

    """
    structure_keys = ["Bridge", "Extended"]
    structure_labels = ["Parallel\nBeta-Sheet", "Anti-parallel\nBeta-sheet"]

    genotypes = list(genotype_data.keys())
    color_map = {"WT": "#343436", "D239N": "#ff0202", "K637E": "#0e8400"}

    fig, ax = plt.subplots(figsize=(4, 6), dpi=300)

    bar_width = 0.025
    bar_spacing = 0.015
    group_spacing = 0.15

    num_structures = len(structure_keys)
    num_genotypes = len(genotypes)
    group_width = num_genotypes * bar_width + (num_genotypes - 1) * bar_spacing
    x = np.arange(num_structures) * (group_width + group_spacing)

    for i, genotype in enumerate(genotypes):
        values = [genotype_data[genotype].get(struct, 0.0) for struct in structure_keys]
        positions = x + i * (bar_width + bar_spacing)

        ax.bar(positions, values, width=bar_width,
               color=color_map.get(genotype, "#666666"),
               edgecolor="black",
               linewidth=1.5,
               alpha=0.7,
               label=genotype)

        for pos, val in zip(positions, values):
            ax.text(pos, val + 1, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, rotation=90)

    ax.set_xticks(x + (group_width - bar_width) / 2)
    ax.set_xticklabels(structure_labels, rotation=45, ha="right")

    ax.set_ylabel("Percent of Simulations", labelpad=18)
    ax.set_xlabel("Beta-Sheet Type", labelpad=18)
    ax.set_ylim(0, 5)

    ax.set_title(title, pad=55)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(pad=2.0)
    plt.show()

# ================================================================
# RESIDUE-LEVEL ANOVA (SEPARATE FOLDERS)
# ================================================================

def _read_sum_dat(path, structure_col):
    """
    Read a single .sum.dat file and extract the residue column
    and the selected structure column.

    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    df = df.rename(columns={"#Residue": "Residue"})
    df["Residue"] = pd.to_numeric(df["Residue"], errors="coerce")
    df[structure_col] = pd.to_numeric(df[structure_col], errors="coerce")

    return df[["Residue", structure_col]].dropna()

def _collect_group_means_three(folder_paths, structure_col):
    """
    Collect per-residue means for each genotype across all replicates.

    """
    group_maps = {}

    for label, folder in folder_paths.items():
        files = [f for f in os.listdir(folder) if f.endswith("sum.dat")]
        values = {}

        for f in files:
            df = _read_sum_dat(os.path.join(folder, f), structure_col)
            grouped = df.groupby("Residue")[structure_col].mean()

            for r, v in grouped.items():
                values.setdefault(int(r), []).append(float(v))

        group_maps[label] = values

    return group_maps

def compute_residue_anova(folder_paths, structure_col="Extended",
                          pval_threshold=0.05):
    """
    Perform one-way ANOVA for each residue across WT, D239N, and K637E.

    """
    group_maps = _collect_group_means_three(folder_paths, structure_col)
    all_residues = sorted(set().union(*[set(m.keys()) for m in group_maps.values()]))

    rows = []
    for r in all_residues:
        vals = {label: group_maps[label].get(r, []) for label in folder_paths}

        nonempty = [g for g in vals.values() if len(g) > 0]
        pval = f_oneway(*nonempty).pvalue if len(nonempty) >= 2 else np.nan

        row = {"Residue": r, "pval": pval}
        for label, g in vals.items():
            row[f"{label}_mean"] = np.mean(g) * 100 if g else np.nan
            row[f"{label}_sem"] = np.std(g, ddof=1) * 100 / np.sqrt(len(g)) if len(g) >= 2 else np.nan
            row[f"{label}_vals"] = [v * 100 for v in g]

        rows.append(row)

    full_df = pd.DataFrame(rows)
    sig_df = full_df[full_df["pval"] < pval_threshold]

    return full_df, sig_df

def plot_significant_residues_anova(folder_paths,
                                    structure_col="Extended",
                                    pval_threshold=0.05,
                                    title=None):
    """
    Plot only the residues that show significant differences across genotypes
    based on one-way ANOVA.

    """
    full_df, sig_df = compute_residue_anova(folder_paths, structure_col, pval_threshold)

    if sig_df.empty:
        print("No significant residues.")
        return full_df

    genotypes = list(folder_paths.keys())
    x = np.arange(len(sig_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(4, 6))

    for i, label in enumerate(genotypes):
        means = sig_df[f"{label}_mean"].values
        sems = sig_df[f"{label}_sem"].values
        positions = x + (i - 1) * width

        ax.bar(positions, means, width, yerr=sems, capsize=4, label=label, alpha=0.8)

        for j, pos in enumerate(positions):
            vals = sig_df.iloc[j][f"{label}_vals"]
            ax.scatter(np.full(len(vals), pos), vals, s=30, edgecolor="black", zorder=3)

    for i, row in sig_df.iterrows():
        idx = sig_df.index.get_loc(i)
        ymax = np.nanmax([row[f"{g}_mean"] for g in genotypes])
        ax.text(x[idx], ymax + 2.75, f"p={row['pval']:.1e}", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(sig_df["Residue"].astype(int))
    ax.set_xlabel("Residue", labelpad=18)
    ax.set_ylabel("Fraction of Frames (%)", labelpad=18)
    ax.set_ylim(0, 17)
    ax.set_title(title or f"Significant Residues for {structure_col} (ANOVA)", pad=30)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(pad=2.0)
    plt.show()

    return sig_df

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    # Loads structure summaries for each genotype and generates structure-level plots
    base_folder = r"C:\Users\salar\Documents\MAB_Project\Results\cpptraj_loop2_secondary"

    wt = entire_simulation_percentages(os.path.join(base_folder, "wt"), keyword="wt", plot=False)
    d239n = entire_simulation_percentages(os.path.join(base_folder, "d239n"), keyword="d239n", plot=False)
    k637e = entire_simulation_percentages(os.path.join(base_folder, "k637e"), keyword="k637e", plot=False)

    genotype_data = {"WT": wt.to_dict(), "D239N": d239n.to_dict(), "K637E": k637e.to_dict()}

    # Generates structure-level plots
    plot_loop2_organization(genotype_data, title="Loop 2 Organization")
    plot_loop2_organization2(genotype_data, title="Loop 2 Beta-Sheet Organization")

    # Runs residue-level ANOVA and plots significant residues
    folder_paths = {
        "WT": os.path.join(base_folder, "wt"),
        "D239N": os.path.join(base_folder, "d239n"),
        "K637E": os.path.join(base_folder, "k637e"),
    }

    # Computes ANOVA and plots significant residues for the "Extended" structure type (anti-parallel beta-sheets)
    sig_residues = plot_significant_residues_anova(
        folder_paths,
        structure_col="Extended",
        pval_threshold=0.05,
        title="Anti-parallel Beta-Sheets in Loop 2",
    )
