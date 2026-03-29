import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from matplotlib.lines import Line2D

# ======================================================================================================
# GLOBAL PLOTTING STYLE — ensures ALL plots look consistent
# ======================================================================================================

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
    "axes.labelpad": 20, 
})

# ======================================================================================================
# FILE PATHS
# ======================================================================================================
csv_d239n = r"C:\Users\salar\Documents\MAB_Project\Results\contacts\output\contact_diffs_all_d239n.csv"
csv_k637e = r"C:\Users\salar\Documents\MAB_Project\Results\contacts\output\contact_diffs_all_k637e.csv"

# ======================================================================================================
# DOMAIN LABELS FOR BRACKET ANNOTATION
# ======================================================================================================
domain_labels = {
    "Upper 50kDa Domain": ["215-231", "266-453", "604-621"],
    "Lower 50kDa Domain": ["472-566","578-590", "645-665"],
    "S1": ["234-244"],
    "SH1": ["672-685"],
    "Purine Loop": ["126-131"],
    "P Loop": ["179-183"],
    "N Terminal": ["113-123", "666-671", "170-175", "454-460","244-265"],
    "Loop 2": ["624-646"],
    "Loop 3": ["567-577"],
    "S2": ["462-472"]
}

# ======================================================================================================
# 1. LOADING + NORMALIZATION HELPERS
# ======================================================================================================

def load_contact_csv(path):
    """
    Load a contact CSV exactly as-is.
    """
    return pd.read_csv(path)

def normalize_pair(pair):
    """
    Ensure contact pairs are always formatted as smaller-larger.
    """
    a, b = map(int, pair.split('-'))
    return f"{min(a,b)}-{max(a,b)}"

def normalize_contact_column(df):
    """
    Apply normalization to the '#Contact' column.
    """
    df['#Contact'] = df['#Contact'].astype(str).apply(normalize_pair)
    return df

def reorder_pair_with_anchor(pair, anchor):
    """
    Ensure the anchor residue is always first in the pair.
    """
    a, b = map(int, pair.split('-'))
    if a == anchor:
        return f"{a}-{b}"
    if b == anchor:
        return f"{b}-{a}"
    return pair

def normalize_domain_pairs(pair, domain_focus):
    """
    Reorder a contact pair so the domain residue is always first.
    """
    a, b = map(int, pair.split('-'))
    ranges = domain_labels[domain_focus]

    domain_res = set()
    for r in ranges:
        low, high = map(int, r.split('-'))
        domain_res.update(range(low, high + 1))

    if a in domain_res:
        return f"{a}-{b}"
    if b in domain_res:
        return f"{b}-{a}"
    return f"{a}-{b}"

# ======================================================================================================
# 2. FILTERING FUNCTIONS
# ======================================================================================================

def filter_contacts(df, residue_range, p_val=0.05):
    """
    Filter rows where the contact pair contains ANY residue in the given range
    AND p-value < p_val.
    """
    low, high = residue_range
    filtered = []

    for r in range(low, high + 1):
        pattern = rf'\b{r}\b'
        subset = df[df['#Contact'].str.contains(pattern, regex=True) & (df['pval'] < p_val)]
        filtered.append(subset)

    if filtered:
        return pd.concat(filtered).drop_duplicates()
    return pd.DataFrame()

# ======================================================================================================
# 3. GENOTYPE DICTIONARY BUILDER
# ======================================================================================================

def compute_averages(df):
    """
    Compute WT and MUT averages for each row.
    """
    df["WTavg"] = df[["Run1", "Run2", "Run3"]].mean(axis=1)
    df["MUTavg"] = df[["Run1.1", "Run2.1", "Run3.1"]].mean(axis=1)
    return df

def build_genotype_dicts(df239, df637, df239_filtered, df637_filtered):
    """
    Build WT, D239N, and K637E dictionaries for plotting.
    """
    for df in [df239, df637, df239_filtered, df637_filtered]:
        df['#Contact'] = df['#Contact'].apply(normalize_pair)

    all_pairs = sorted(
        set(df239_filtered['#Contact']) | set(df637_filtered['#Contact']),
        key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]))
    )

    df239 = compute_averages(df239)
    df637 = compute_averages(df637)

    wt239 = dict(zip(df239['#Contact'], df239['WTavg']))
    wt637 = dict(zip(df637['#Contact'], df637['WTavg']))
    m239 = dict(zip(df239['#Contact'], df239['MUTavg']))
    m637 = dict(zip(df637['#Contact'], df637['MUTavg']))

    wt_dict = {}
    for k in all_pairs:
        if k in wt239 and k in wt637:
            wt_dict[k] = (wt239[k] + wt637[k]) / 2
        elif k in wt239:
            wt_dict[k] = wt239[k]
        elif k in wt637:
            wt_dict[k] = wt637[k]

    genotype_data = {
        "WT": wt_dict,
        "D239N": {k: m239[k] for k in all_pairs if k in m239},
        "K637E": {k: m637[k] for k in all_pairs if k in m637}
    }

    return genotype_data

# ======================================================================================================
# 4. PLOTTING FUNCTIONS
# ======================================================================================================

def plot_loop2_organization(genotype_data, title="Loop 2 Contact Analysis"):
    """
    Compare WT, D239N, and K637E contact frequencies for Loop 2.
    Bars are grouped by contact pair and colored by genotype.
    """

    # Collect all contact pairs across genotypes
    structure_keys = sorted(
        set().union(*[genotype_data[g].keys() for g in genotype_data]),
        key=lambda x: int(x.split('-')[1])
    )

    genotypes = list(genotype_data.keys())
    color_map = {"WT": "#343436", "D239N": "#4101ff", "K637E": "#0e8400"}

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    bar_width = 0.025
    bar_spacing = 0.015
    group_spacing = 0.15

    num_structures = len(structure_keys)
    num_genotypes = len(genotypes)

    group_width = num_genotypes * bar_width + (num_genotypes - 1) * bar_spacing
    x = np.arange(num_structures) * (group_width + group_spacing)

    # Draw bars for each genotype
    for i, genotype in enumerate(genotypes):
        values = [genotype_data[genotype].get(struct, 0) for struct in structure_keys]
        positions = x + i * (bar_width + bar_spacing)

        ax.bar(
            positions, values, width=bar_width,
            color=color_map[genotype], edgecolor="black", linewidth=1.8,
            label=genotype
        )

        # Add labels above bars
        for pos, val in zip(positions, values):
            ax.text(pos, val + 1, f"{val:.2f}%", ha="center", va="bottom",
                    fontsize=10, rotation=90)

    # X-axis formatting
    ax.set_xticks(x + (group_width - bar_width) / 2)
    ax.set_xticklabels(structure_keys, rotation=45, ha="right",
                       fontsize=16, fontweight='semibold')

    ax.set_ylabel("Percent of Simulations", fontsize=16, fontweight='semibold')
    ax.set_yticks(np.arange(0, 51, 10))
    ax.tick_params(axis='both', labelsize=12)

    # Legend
    legend_elements = [
        Line2D([0], [0], linestyle='None', label='Genotype'),
        *[
            Line2D([0], [0], marker='s', color='black',
                   markerfacecolor=color_map[g], markeredgecolor='black',
                   markeredgewidth=1.5, markersize=12, linestyle='None', label=g)
            for g in genotypes
        ]
    ]

    ax.legend(handles=legend_elements, loc="upper center",
              ncol=len(genotypes) + 1, frameon=False, fontsize=12,
              prop={'weight': 'bold'}, handletextpad=0.3, columnspacing=0.8)

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

# ======================================================================================================
# 5. RESIDUE-LEVEL PLOT
# ======================================================================================================

def plot_single_residue_contacts(df, residue, title=None, domain_groups=domain_labels):
    """
    Plot all WT vs MUT contact pairs involving a given residue.
    Includes:
        - WT/MUT means + SEM
        - replicate scatter
        - p-values
        - domain brackets under the x-axis
    """

    if title is None:
        title = f"All Contacts Involving Residue {residue}"

    df = df.copy()

    # Extract only contacts involving the residue
    df_res = df[
        df['#Contact'].str.contains(f"{residue}-") |
        df['#Contact'].str.contains(f"-{residue}")
    ].copy()

    if df_res.empty:
        print(f"No contacts found involving residue {residue}.")
        return

    # Reorder so the anchor residue is always first
    df_res['plot_label'] = df_res['#Contact'].apply(
        lambda p: reorder_pair_with_anchor(p, residue)
    )

    # Sort by the “other” residue
    df_res['other'] = df_res['plot_label'].apply(lambda p: int(p.split('-')[1]))
    df_res = df_res.sort_values('other').reset_index(drop=True)

    # Compute WT/MUT means + SEM
    wt_cols = ["Run1", "Run2", "Run3"]
    mut_cols = ["Run1.1", "Run2.1", "Run3.1"]

    wt_values = df_res[wt_cols].values
    mut_values = df_res[mut_cols].values

    wt_means = wt_values.mean(axis=1)
    mut_means = mut_values.mean(axis=1)

    wt_sems = sem(wt_values, axis=1)
    mut_sems = sem(mut_values, axis=1)

    # Plotting
    x = np.arange(len(df_res))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

    ax.bar(x - width/2, wt_means, width,
           yerr=wt_sems, capsize=5,
           label="WT", color="skyblue", alpha=0.8)

    ax.bar(x + width/2, mut_means, width,
           yerr=mut_sems, capsize=5,
           label="MUT", color="salmon", alpha=0.8)

    # Add replicate scatter points
    for i in range(len(df_res)):
        ax.scatter(
            np.full(3, x[i] - width/2) + np.random.uniform(-0.05, 0.05, 3),
            wt_values[i],
            color="blue", zorder=10
        )
        ax.scatter(
            np.full(3, x[i] + width/2) + np.random.uniform(-0.05, 0.05, 3),
            mut_values[i],
            color="red", edgecolor="black", zorder=10
        )

    # Add p-values
    for i, pval in enumerate(df_res["pval"]):
        y = max(wt_means[i] + wt_sems[i], mut_means[i] + mut_sems[i]) + 10
        label = f"p={pval:.3f}" if pval < 0.05 else ""
        ax.text(i, y, label, ha="center", va="bottom", fontsize=10)

    # Domain brackets under the x-axis
    y_bracket = -0.08
    y_label = -0.10
    pad = width * 1.2

    index_to_residue = {i: df_res['other'].iloc[i] for i in range(len(df_res))}

    for domain, ranges in domain_groups.items():

        # Collect all residues in this domain
        domain_res = set()
        for r in ranges:
            low, high = map(int, r.split('-'))
            domain_res.update(range(low, high + 1))

        # Find x positions where the second residue belongs to this domain
        domain_indices = [
            idx for idx, resnum in index_to_residue.items()
            if resnum in domain_res
        ]

        if not domain_indices:
            continue

        x_start = min(domain_indices)
        x_end   = max(domain_indices)
        x_center = (x_start + x_end) / 2

        # Horizontal bracket line
        ax.plot([x_start - pad, x_end + pad],
                [y_bracket, y_bracket],
                transform=ax.get_xaxis_transform(),
                color='black', linewidth=1.5, clip_on=False)

        # Left tick
        ax.plot([x_start - pad, x_start - pad],
                [y_bracket, y_bracket + 0.02],
                transform=ax.get_xaxis_transform(),
                color='black', linewidth=1.5, clip_on=False)

        # Right tick
        ax.plot([x_end + pad, x_end + pad],
                [y_bracket, y_bracket + 0.02],
                transform=ax.get_xaxis_transform(),
                color='black', linewidth=1.5, clip_on=False)

        # Domain label
        ax.text(
            x_center, y_label, domain,
            ha="center", va="top",
            fontsize=10,
            transform=ax.get_xaxis_transform(),
            clip_on=False
        )

    # Final formatting
    ax.set_xticks(x)
    ax.set_xticklabels(df_res['plot_label'], ha="center")
    ax.set_xlabel("Residue-Residue Contacts", fontweight="bold", labelpad=50)
    ax.set_ylim(0, 1.5 * max(wt_means + mut_means))
    ax.set_ylabel("Average Percentage of Frames (%)", fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ======================================================================================================
# 6. DOMAIN-LEVEL SIGNIFICANT CONTACT PLOT
# ======================================================================================================

def plot_domain_contacts(df_d239n, domain_focus, p_val=0.05):
    """
    Plot all significant WT vs MUT contacts for a given domain.
    """
    ranges = domain_labels[domain_focus]
    domain_groups = domain_labels

    filtered = []
    for r in ranges:
        low, high = map(int, r.split('-'))
        df_f = filter_contacts(df_d239n, (low, high), p_val)
        if not df_f.empty:
            filtered.append(df_f)

    if not filtered:
        print(f"No significant contacts found for {domain_focus}.")
        return

    df_dom = pd.concat(filtered).copy()

    df_dom['Contact'] = df_dom["#Contact"].apply(
        lambda p: normalize_domain_pairs(p, domain_focus)
    )

    df_dom['other'] = df_dom['Contact'].apply(lambda p: int(p.split('-')[1]))
    df_dom = df_dom.sort_values('other').reset_index(drop=True)

    wt_cols = ["Run1", "Run2", "Run3"]
    mut_cols = ["Run1.1", "Run2.1", "Run3.1"]

    wt_values = df_dom[wt_cols].values
    mut_values = df_dom[mut_cols].values

    wt_means = wt_values.mean(axis=1)
    mut_means = mut_values.mean(axis=1)

    wt_sems = sem(wt_values, axis=1)
    mut_sems = sem(mut_values, axis=1)

    x = np.arange(len(df_dom))
    width = 0.35
    fig_width = max(16, len(df_dom) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 10), dpi=200)

    ax.bar(x - width/2, wt_means, width, yerr=wt_sems, capsize=5,
           label="WT", color="skyblue")

    ax.bar(x + width/2, mut_means, width, yerr=mut_sems, capsize=5,
           label="MUT", color="salmon")

    for i in range(len(df_dom)):
        ax.scatter(x[i] - width/2 + np.random.uniform(-0.05, 0.05, 3),
                   wt_values[i], color="blue", zorder=10)
        ax.scatter(x[i] + width/2 + np.random.uniform(-0.05, 0.05, 3),
                   mut_values[i], color="red", zorder=10)

    # Domain brackets
    y_bracket = -0.08
    y_label   = -0.10
    pad = width * 1.2

    index_to_residue = {i: df_dom['other'].iloc[i] for i in range(len(df_dom))}

    for domain, ranges in domain_groups.items():

        # Build set of all residues in this domain
        domain_res = set()
        for r in ranges:
            low, high = map(int, r.split('-'))
            domain_res.update(range(low, high + 1))

        # Collect x positions where the second residue falls in this domain
        domain_indices = [
            idx for idx, res in index_to_residue.items()
            if res in domain_res
        ]

        if not domain_indices:
            continue

        x_start = min(domain_indices)
        x_end   = max(domain_indices)
        x_center = (x_start + x_end) / 2

        # Draw bracket line
        ax.plot([x_start - pad, x_end + pad],
                [y_bracket, y_bracket],
                transform=ax.get_xaxis_transform(),
                color='black', linewidth=1.5, clip_on=False)

        # Left tick
        ax.plot([x_start - pad, x_start - pad],
                [y_bracket, y_bracket + 0.01],
                transform=ax.get_xaxis_transform(),
                color='black', linewidth=1.5, clip_on=False)

        # Right tick
        ax.plot([x_end + pad, x_end + pad],
                [y_bracket, y_bracket + 0.01],
                transform=ax.get_xaxis_transform(),
                color='black', linewidth=1.5, clip_on=False)

        # Domain label
        ax.text(
            x_center, y_label, domain,
            ha="center", va="top",
            fontsize=11,
            transform=ax.get_xaxis_transform(),
            clip_on=False
        )

    # Final formatting
    ax.set_xticks(x)
    ax.set_xticklabels(df_dom['Contact'], ha="center", fontsize=12)
    ax.set_xlabel("Residue-Residue Contacts", fontweight="bold", labelpad=65)
    ax.set_ylabel("Average Percentage of Frames (%)", fontweight="bold")
    ax.set_ylim(0, 1.25 * max(wt_means.max(), mut_means.max()))
    ax.set_title(f"Significant Contacts in {domain_focus}",
                 fontsize=20, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


# ======================================================================================================
# 7. MAIN EXECUTION BLOCK
# ======================================================================================================

def main():
    """
    Full execution pipeline for contact‑difference analysis.
    Loads both genotype CSVs, normalizes contact pairs, filters Loop 2 contacts,
    builds genotype dictionaries, and generates all plots.
    """

    # Load raw CSVs
    print("Loading CSV files...")
    df_d239n = load_contact_csv(csv_d239n)
    df_k637e = load_contact_csv(csv_k637e)

    # Normalize contact pair formatting
    print("Normalizing contact pairs...")
    df_d239n = normalize_contact_column(df_d239n)
    df_k637e = normalize_contact_column(df_k637e)

    # Filter Loop 2 contacts for each genotype
    print("Filtering Loop 2 contacts...")
    d239n_loop2_df = filter_contacts(df_d239n, (621, 646), p_val=0.05)
    k637e_loop2_df = filter_contacts(df_k637e, (621, 646), p_val=0.05)

    # Build genotype dictionaries for Loop 2 organization plot
    print("Building genotype dictionaries...")
    genotype_data = build_genotype_dicts(
        df_d239n, df_k637e,
        d239n_loop2_df, k637e_loop2_df
    )

    # Plot Loop 2 organization (WT vs D239N vs K637E)
    print("Plotting Loop 2 organization...")
    plot_loop2_organization(genotype_data, title="Loop 2 Contact Analysis")

    # Plot residue‑specific WT vs MUT comparisons
    print("Plotting residue‑level analyses...")
    residues_to_plot = [239, 235, 674, 677, 237, 466, 227, 589]

    for res in residues_to_plot:
        plot_single_residue_contacts(df_d239n, res)

    # Plot domain‑level significant contacts
    print("Plotting domain‑level significant contacts...")
    domains_to_plot = [
        "Upper 50kDa Domain",
        "Lower 50kDa Domain",
        "S1",
        "SH1",
        "Purine Loop",
        "P Loop"
    ]

    for domain in domains_to_plot:
        plot_domain_contacts(df_d239n, domain, p_val=0.05)


# ======================================================================================================
# Main Execution
# ======================================================================================================

if __name__ == "__main__":
    main()
