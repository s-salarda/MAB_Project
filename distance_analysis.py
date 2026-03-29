# ================================================================
# Distance Analysis based on the COM
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
from scipy.stats import ttest_ind

# ================================================================
# GLOBAL PLOTTING STYLE
# ================================================================

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.frameon": True,
    "legend.fontsize": 12,
    "figure.dpi": 200,
})

# ================================================================
# BUILD TIDY DATAFRAME (FRAME-LEVEL)
# ================================================================

def build_tidy_dataframe(directory):
    """
    Build a tidy DataFrame from all COM-based .dat distance files.

    Columns:
        State      : 'wt' or 'mut'
        Simulation : 1, 2, or 3
        Frame      : time index
        Distance   : COM distance (Å)
        Domain     : structural domain (e.g., 'S1_Ohelix')
        File       : filename
    """
    directory = Path(directory)
    rows = []

    for file in directory.glob("*.dat"):
        name = file.stem
        parts = name.split("_")

        domain = parts[0] + "_" + parts[1]
        type_sim = parts[2]              # e.g., "wt1"
        state = re.match(r"(wt|mut)", type_sim).group(1)
        sim = int(re.search(r"\d+", type_sim).group(0))

        # Load full time series (frame, distance)
        data = np.loadtxt(file)

        for frame, dist in data:
            rows.append({
                "State": state,
                "Simulation": sim,
                "Frame": int(frame),
                "Distance": float(dist),
                "Domain": domain,
                "File": file.name
            })

    return pd.DataFrame(rows)

# ================================================================
# WRITE TIDY DF TO TXT
# ================================================================

def write_tidy_to_txt(df, out_path):
    """Save tidy DataFrame to a tab-delimited text file."""
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Tidy distance data written to: {out_path}")

# ================================================================
# PER-FILE MEAN DF (SIMULATION-LEVEL)
# ================================================================

def process_distance_files(directory):
    """
    Extract per-file mean distances for each simulation.
    Produces the same structure your original plotting functions expect.
    """
    directory = Path(directory)
    records = []

    for file in directory.glob("*.dat"):
        name = file.stem
        parts = name.split("_")

        domain = parts[0] + "_" + parts[1]
        type_sim = parts[2]
        type_ = re.match(r"(wt|mut)", type_sim).group(1)
        sim = int(re.search(r"\d+", type_sim).group(0))

        distances = np.loadtxt(file, usecols=1)
        file_mean = distances.mean()

        records.append({
            "domain": domain,
            "type": type_,
            "sim": sim,
            "file_mean": file_mean
        })

    df = pd.DataFrame(records)

    # Add group-level mean for convenience
    group_means = (
        df.groupby(["domain", "type"])["file_mean"]
        .mean()
        .reset_index()
        .rename(columns={"file_mean": "group_mean"})
    )

    return df.merge(group_means, on=["domain", "type"], how="left")

# ================================================================
# SECTION DEFINITIONS
# ================================================================

SECTIONS = {
    "ADP": ["ADP_SH1", "ADP_S1", "ADP_Ploop", "ADP_Purine"],
    "Ploop-Phosphate": ["Ploop_Pi@P"],
    "Switch1": ["S1_SH1", "S1_U50", "S1_L50", "S1_Ohelix", "S1_Relay"],
    "Cleft": ["S1_S2", "U50_L50", "HLH_L2", "L3_L2", "Ohelix_Relay"],
    "Residue to Residue 239": ["239_320", "239_321", "239_323", "239_679"],
    "Phosphate Backdoor": ["238_466", "Pi_238", "Pi_466"],
}

# ================================================================
# SPLIT INTO SECTION DATAFRAMES
# ================================================================

def split_into_section_dataframes(sections, directory):
    """
    Return a dictionary of DataFrames, one per structural section.
    """
    df = process_distance_files(directory)

    domain_to_section = {
        domain: section
        for section, domains in sections.items()
        for domain in domains
    }

    df["section"] = df["domain"].map(domain_to_section)

    return {
        section: df[df["section"] == section].copy()
        for section in sections
    }

# ================================================================
# SIMULATION-LEVEL WT/MUT STATS
# ================================================================

def compute_stats(df, domain):
    """
    Compute WT vs MUT statistics using simulation-level means.
    This preserves your original bar-plot behavior.
    """
    wt_vals = df[(df["domain"] == domain) & (df["type"] == "wt")]["file_mean"].astype(float)
    mut_vals = df[(df["domain"] == domain) & (df["type"] == "mut")]["file_mean"].astype(float)

    wt_mean = wt_vals.mean()
    mut_mean = mut_vals.mean()
    wt_sem = wt_vals.sem()
    mut_sem = mut_vals.sem()

    if len(wt_vals) == 3 and len(mut_vals) == 3:
        pval = ttest_ind(wt_vals, mut_vals).pvalue
    else:
        pval = np.nan

    return wt_vals, mut_vals, wt_mean, mut_mean, wt_sem, mut_sem, pval

# ================================================================
# SECTION PLOT (ORIGINAL PERFECT LOOK)
# ================================================================

def plot_section(section_name, section_df):
    """
    WT vs MUT bar plot for all domains in a section.
    Uses simulation-level means (3 WT sims, 3 MUT sims).
    """
    section_df['domain'] = section_df['domain'].str.replace('_', '-')
    domains = section_df["domain"].unique()
    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    wt_means, mut_means = [], []
    wt_sems, mut_sems = [], []
    pvals = []
    stats_dict = {}

    for domain in domains:
        wt_vals, mut_vals, wt_mean, mut_mean, wt_sem, mut_sem, pval = compute_stats(section_df, domain)
        stats_dict[domain] = (wt_vals, mut_vals)

        wt_means.append(wt_mean)
        mut_means.append(mut_mean)
        wt_sems.append(wt_sem)
        mut_sems.append(mut_sem)
        pvals.append(pval)

    # Bars
    ax.bar(x - width/2, wt_means, width, yerr=wt_sems,
           label="WT", color="skyblue", capsize=5)
    ax.bar(x + width/2, mut_means, width, yerr=mut_sems,
           label="Mut", color="salmon", capsize=5)

    # Scatter points (3 per group)
    for i, domain in enumerate(domains):
        wt_vals, mut_vals = stats_dict[domain]
        ax.scatter(np.repeat(x[i] - width/2, len(wt_vals)), wt_vals, color="blue", s=40)
        ax.scatter(np.repeat(x[i] + width/2, len(mut_vals)), mut_vals, color="red", s=40)

    # p-values
    for i, p in enumerate(pvals):
        if p < 0.05:
            y_max = max(wt_means[i], mut_means[i])
            ax.text(x[i], y_max * 1.15, f"p={p:.3f}", ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_xlabel("Residue to Residue Pairs", fontweight="bold")
    ax.set_ylabel("Average Distance (Å)", fontweight="bold")
    ax.set_title(f"Distance analysis of {section_name}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(wt_means + mut_means) * 1.4)
    ax.legend()

    plt.tight_layout()
    plt.show()

# ================================================================
# TIME SERIES PLOT (NO SMOOTHING, WITH SEM)
# ================================================================

def plot_time_series(df, domain):
    """
    Plot WT vs MUT mean distance over time with SEM shading.

    """
    sub = df[df["Domain"] == domain]

    # Compute per-frame mean and SEM across simulations
    wt_group = sub[sub["State"] == "wt"].groupby(["Frame", "Simulation"])["Distance"].mean()
    mut_group = sub[sub["State"] == "mut"].groupby(["Frame", "Simulation"])["Distance"].mean()

    wt_stats = wt_group.groupby("Frame").agg(["mean", "sem"])
    mut_stats = mut_group.groupby("Frame").agg(["mean", "sem"])

    plt.figure(figsize=(10, 5))

    # WT
    plt.plot(wt_stats.index, wt_stats["mean"], color="blue", label="WT")
    plt.fill_between(
        wt_stats.index,
        wt_stats["mean"] - wt_stats["sem"],
        wt_stats["mean"] + wt_stats["sem"],
        color="blue",
        alpha=0.2
    )

    # MUT
    plt.plot(mut_stats.index, mut_stats["mean"], color="red", label="MUT")
    plt.fill_between(
        mut_stats.index,
        mut_stats["mean"] - mut_stats["sem"],
        mut_stats["mean"] + mut_stats["sem"],
        color="red",
        alpha=0.2
    )

    plt.title(f"{domain} — WT vs MUT Distance Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Distance (Å)")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

# ================================================================
# MAIN EXECUTION
# ================================================================

folder_path = r"C:\Users\salar\MAB_project\Distance Analysis"

# Build tidy DF (frame-level)
tidy_df = build_tidy_dataframe(folder_path)
write_tidy_to_txt(tidy_df, folder_path + "/distances_tidy.txt")

# Build per-file DF (simulation-level)
section_dfs = split_into_section_dataframes(SECTIONS, folder_path)

# Section-level WT vs MUT plots
for section_name, section_df in section_dfs.items():
    plot_section(section_name, section_df)

# Time-series plots for a chosen section (example: Switch1)
for domain in SECTIONS["Switch1"]:
    plot_time_series(tidy_df, domain)
