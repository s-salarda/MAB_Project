import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
import re
from scipy.stats import ttest_ind

# -----------------------------------------------------------------------
# 1. Load raw cpptraj structure files
# -----------------------------------------------------------------------
def load_structure_files(folder_path, keyword=None):
    """
    Loads all *.sum.dat structure files in a folder and returns a dictionary
    of DataFrames, one per simulation.

    Parameters
    ----------
    folder_path : str
        Directory containing cpptraj *.sum.dat files.
    keyword : str or None
        Optional substring to filter filenames.

    Returns
    -------
    dict
        { filename : DataFrame }
    """
    dat_files = [f for f in os.listdir(folder_path) if f.endswith("sum.dat")]
    if keyword:
        dat_files = [f for f in dat_files if keyword in f]

    dfs = {}
    for file in dat_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, delimiter=r"\s+", header=None)

        # First row is header
        df.columns = df.iloc[0]
        df.index = df.iloc[:, 0]
        df = df.iloc[1:].reset_index(drop=True)

        # Rename cpptraj structure columns
        df = df.rename(columns={
            "Extended": "Anti-parallel Beta-sheet",
            "Bridge": "Parallel Beta-Sheet",
            "3-10": "3-10 Helix",
            "Alpha": "Alpha helix",
            "Pi": "Pi (3-14) helix",
        })

        dfs[file] = df

    return dfs


# -----------------------------------------------------------------------
# 2. Compute per-simulation structure percentages
# -----------------------------------------------------------------------
def entire_simulation_percentages(folder_path, keyword=None, plot=False):
    """
    Computes structure percentages for each simulation in a folder.

    Returns
    -------
    combined : DataFrame
        Rows = simulations, Columns = structure percentages.
    avg_ordered : Series
        Mean structure percentages across simulations.
    """
    dfs = load_structure_files(folder_path, keyword)

    desired_order = [
        "None", "Parallel Beta-Sheet", "Anti-parallel Beta-sheet",
        "3-10 Helix", "Alpha helix", "Pi (3-14) helix", "Turn", "Bend"
    ]

    avg_results = []

    for file, df in dfs.items():
        df_no_res = df.iloc[1:].reset_index(drop=True)
        df_numeric = df_no_res.apply(pd.to_numeric, errors="coerce")

        # Drop residue column
        exclude_cols = {c for c in df_numeric.columns if str(c).lower().strip() in {"#residue", "residue"}}
        df_struct = df_numeric.drop(columns=list(exclude_cols), errors="ignore")

        df_struct = df_struct.dropna(axis=1, how="all")

        col_means = df_struct.mean() * 100
        none_value = max(0, 100 - col_means.sum())
        col_means["None"] = none_value

        ordered = col_means.reindex([c for c in desired_order if c in col_means.index])
        avg_results.append(ordered)

    combined = pd.concat(avg_results, axis=1).T
    avg_ordered = combined.mean().reindex(desired_order)
    print (avg_ordered)

    return combined, avg_ordered


# -----------------------------------------------------------------------
# 3. Build tidy DataFrame (cleanest architecture)
# -----------------------------------------------------------------------
def genotype_df():
    """
    Loads WT, D239N, and K637E simulations and returns:
        combined_dict : dict of DataFrames (all simulations)
        avg_dict : dict of Series (averages)
    """
    genotypes = ["wt", "d239n", "k637e"]
    base_path = r"C:\Users\salar\Desktop\xbc_pps_simulations\cpptraj_analysis_MAB\cpptraj_loop2_secondary"

    combined_dict = {}
    avg_dict = {}

    for g in genotypes:
        folder_path = f"{base_path}\\{g}"
        combined, avg = entire_simulation_percentages(folder_path, keyword=g, plot=False)
        combined_dict[g.upper()] = combined
        avg_dict[g.upper()] = avg

    return combined_dict, avg_dict


def tidy_from_combined(combined_dict):
    """
    Converts combined_dict into a single tidy DataFrame.

    Returns
    -------
    tidy_df : DataFrame
        Columns: genotype, simulation_id, structure columns, total_beta
    """
    records = []

    for genotype, df in combined_dict.items():
        for sim_idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict["genotype"] = genotype
            row_dict["simulation_id"] = sim_idx
            row_dict["total_beta"] = (
                row_dict.get("Parallel Beta-Sheet", 0) +
                row_dict.get("Anti-parallel Beta-sheet", 0)
            )
            records.append(row_dict)

    tidy_df = pd.DataFrame(records)

    # Reorder columns
    cols = ["genotype", "simulation_id"] + \
           [c for c in tidy_df.columns if c not in ["genotype", "simulation_id", "total_beta"]] + \
           ["total_beta"]

    return tidy_df[cols]


# -----------------------------------------------------------------------
# 4. Compute stats (mean + SEM)
# -----------------------------------------------------------------------
def compute_stats(df, structure="total_beta"):
    """
    Computes simulation values, mean, and SEM for a given structure.

    Parameters
    ----------
    df : DataFrame
        Tidy DataFrame from tidy_from_combined()
    structure : str
        Structure column name or "total_beta"

    Returns
    -------
    stats_df : DataFrame
        Columns: sim1, sim2, sim3, mean, sem
    """
    genotypes = ["WT", "D239N", "K637E"]

    sim1, sim2, sim3 = [], [], []
    means, sems, pvals = [], [], []

    wt_vals = df[df["genotype"] == "WT"][structure].values
    d239n_vals = df[df["genotype"] == "D239N"][structure].values
    k637e_vals = df[df["genotype"] == "K637E"][structure].values

    for g in genotypes:
        sub = df[df["genotype"] == g]

        vals = sub["total_beta"].values if structure == "total_beta" else sub[structure].values

        sim1.append(vals[0])
        sim2.append(vals[1])
        sim3.append(vals[2])

        means.append(np.mean(vals))
        sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))

        # p-values only for D239N and K637E vs WT 
        if g == "WT": 
            pvals.append(1.0) 
        elif g == "D239N": 
            pvals.append(ttest_ind(wt_vals, d239n_vals, equal_var=False).pvalue) 
        elif g == "K637E": 
            pvals.append(ttest_ind(wt_vals, k637e_vals, equal_var=False).pvalue)

    stats_df = pd.DataFrame(
        {"sim1": sim1, "sim2": sim2, "sim3": sim3, "mean": means, "sem": sems, "pval": pvals},
        index=genotypes
    )

    stats_df.index.name = "Genotype"

    print(stats_df)
    return stats_df


# -----------------------------------------------------------------------
# 5. Plotting Functions (using tidy DF)
# -----------------------------------------------------------------------
def plot_loop2_organization(genotype_data, title="Loop 2 Organization", ax=None):
    # If no axis is passed, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    structure_keys = [
        "None", "Parallel Beta-Sheet", "Anti-parallel Beta-sheet",
        "3-10 Helix", "Alpha helix", "Pi (3-14) helix", "Turn", "Bend"
    ]

    structure_labels = [
        "None", "Parallel\nBeta-Sheet", "Anti-parallel\nBeta-sheet",
        "3-10 Helix", "Alpha helix", "Pi (3-14) helix", "Turn", "Bend"
    ]

    genotypes = list(genotype_data.keys())
    color_map = {"WT": "#343436", "D239N": "#ff0303", "K637E": "#0e8400"}

    bar_width = 0.025
    bar_spacing = 0.015
    group_spacing = 0.15

    num_structures = len(structure_keys)
    num_genotypes = len(genotypes)
    group_width = num_genotypes * bar_width + (num_genotypes - 1) * bar_spacing
    x = np.arange(num_structures) * (group_width + group_spacing)

    for i, genotype in enumerate(genotypes):
        values = [genotype_data[genotype].get(struct, 0) for struct in structure_keys]
        positions = x + i * (bar_width + bar_spacing)

        ax.bar(
            positions, values, width=bar_width,
            color=color_map[genotype], edgecolor="black", linewidth=1.8, alpha = 0.7
        )

        for pos, val in zip(positions, values):
            ax.text(pos, val + 1, f"{val:.2f}%", ha="center", va="bottom",
                    fontsize=10, rotation=90)
            
    
    legend_elements = [
        Line2D([0], [0], linestyle='None', label='Genotype'),
        *[
            Line2D(
                [0], [0],
                marker='s',
                color='black',
                markerfacecolor=color_map[g],
                markeredgecolor='black',
                markeredgewidth=1.5,
                markersize=12,
                linestyle='None',
                label=g,
                alpha = 0.7
            )
            for g in genotypes
        ]
    ]

    leg = ax.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=len(genotypes) + 1,
        frameon=False,
        fontsize=12,
        prop={'weight': 'bold'},
        handletextpad=0.3,
        columnspacing=0.8
    )

    genotype_text = leg.get_texts()[0]
    genotype_text.set_fontsize(12)
    genotype_text.set_fontweight('bold')

    ax.set_xticks(x + (group_width - bar_width) / 2)
    ax.set_xticklabels(structure_labels, rotation=45, ha="right",
                       fontsize=16, fontweight='semibold')

    ax.set_ylabel("Percent of Simulations", fontsize=16, fontweight='semibold', labelpad=10)
    ax.set_yticks(np.arange(0, 81, 10))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='semibold')
    plt.setp(ax.get_yticklabels(), fontweight='semibold')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def plot_loop2_organization2(df, structure="total_beta", ylabel="Percent of Sim", ax=None):
    """
    Bar plot of a single structure with SEM + replicate dots.
    If ax is provided, draw into that axis instead of creating a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,8), dpi=300)

    genotypes = ["WT", "D239N", "K637E"]
    color_map = {"WT": "#343436", "D239N": "#ff0000", "K637E": "#0e8400"}
    color_map2 = {"WT": "#0E0E0E", "D239N": "#150055", "K637E": "#063800"}

    
    stats = compute_stats(df, structure)
    means = stats["mean"].values
    sems = stats["sem"].values
    x = np.arange(len(genotypes))

    ax.bar(
        x, means, yerr=sems,
        color=[color_map[g] for g in genotypes],
        edgecolor="black", linewidth=1.5, capsize=6, width=0.3, alpha = 0.7
    )

    # Replicate dots
    for i, g in enumerate(genotypes):
        vals = df[df["genotype"] == g][structure].values
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
        ax.scatter(x[i] + jitter, vals, color=color_map2[g], 
                   edgecolor="black", s=20, alpha=0.9, zorder=10)
        
    # p-value annotations if p<0.05
    for i, p in enumerate(stats["pval"].values):
        if p < 0.05:
            ax.text(x[i], means[i] + sems[i] + 0.25,
                    f"*", ha="center", va="bottom", 
                    fontsize=20, weight = "bold")

    ax.set_xticks(x)
    ax.set_xticklabels(genotypes, fontsize=16, rotation=45, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="semibold", labelpad=10)
    ax.set_ylim(-0.1, 6)
    ax.tick_params(axis='both', labelsize=12)
    plt.setp(ax.get_xticklabels(), fontweight='semibold')
    plt.setp(ax.get_yticklabels(), fontweight='semibold')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


import matplotlib.gridspec as gridspec

def combine_plots(df, avg_dict, structure1="total_beta", structure2="Alpha helix"):
    """
    Creates a 3‑panel figure with independent axes and different subplot sizes:
    - Panel 1: Full Loop 2 organization (wide)
    - Panel 2: structure1 (medium)
    - Panel 3: structure2 (medium)

    Uses your existing plot functions exactly as written.
    """

    fig = plt.figure(figsize=(20, 6), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    # ax3 = fig.add_subplot(gs[2])

    plot_loop2_organization(avg_dict, ax=ax1)
    plot_loop2_organization2(df, structure1, ax=ax2, ylabel = "Percent of Sim in Anti-parallel β-Sheets")
    # plot_loop2_organization2(df, structure2, ax=ax3, ylabel = "Percent of Sim in Alpha Helix")

    plt.show()


# -----------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------
combined_dict, avg_dict = genotype_df()
df = tidy_from_combined(combined_dict)

plot_loop2_organization(avg_dict, title="Secondary Structure of Loop 2")
plot_loop2_organization2(df,structure="Anti-parallel Beta-sheet", ylabel = "Percent of Sim in Anti-parallel β-Sheet")
plot_loop2_organization2(df,structure="Alpha helix", ylabel = "Percent of Sim in Alpha Helix")
plot_loop2_organization2(df,structure="Parallel Beta-Sheet", ylabel = "Percent of Sim in Parallel β-Sheet")
plot_loop2_organization2(df,structure="total_beta", ylabel = "Percent of Total β-Sheet")
combine_plots(df, avg_dict, structure1= "Anti-parallel Beta-sheet")