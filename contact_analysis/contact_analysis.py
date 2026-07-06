# Native Contact Domain Analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
from scipy.stats import ttest_ind

# Parent folder containing results_wt and results_d239n subfolders
PARENT_FOLDER = r"D:\Projects\MAB_project\Contact_Analysis"  # <-- CHANGE THIS

# Folder where plots will be saved (created automatically if it doesn't exist)
# Set to None to disable saving and only show plots interactively
SAVE_DIR = r"D:\Projects\MAB_project\Contact_Analysis\plots"  # <-- CHANGE OR SET TO None

###############################################################
def pval_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


###############################################################
def save_figure(fig, filename):
    """Save figure to SAVE_DIR if set, then show."""
    if SAVE_DIR is not None:
        save_path = Path(SAVE_DIR)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path / filename}")


###############################################################
def process_native_files(wt_dir, mut_dir):
    records = []

    def load_dir(directory, type_label):
        directory = Path(directory)
        print(f"\nFiles in {directory.name}:")
        for file in directory.glob("*.dat"):
            print(" -", file.name)

        for file in directory.glob("*.dat"):
            name = file.stem
            parts = name.split("_")

            # Expect at least 3 parts: e.g. ADP_SH1_wt1 or ADP_SH1_mut2
            domain = parts[0] + "_" + parts[1]
            type_sim = parts[2]
            sim = re.search(r"\d+", type_sim).group(0)

            native_vals = np.loadtxt(file, usecols=1)
            nonnative_vals = np.loadtxt(file, usecols=2)

            native_vals = native_vals[native_vals != 0]
            nonnative_vals = nonnative_vals[nonnative_vals != 0]

            records.append({
                "domain": domain,
                "type": type_label,
                "sim": sim,
                "native_mean": native_vals.mean() if len(native_vals) > 0 else 0.0,
                "nonnative_mean": nonnative_vals.mean() if len(nonnative_vals) > 0 else 0.0,
            })
            print(f"  Parsed: domain={domain}, type={type_label}, sim={sim}")

    parent = Path(PARENT_FOLDER)
    load_dir(parent / "results_wt",    "wt")
    load_dir(parent / "results_d239n", "mut")

    df = pd.DataFrame(records)

    group_means = (
        df.groupby(["domain", "type"])[["native_mean", "nonnative_mean"]]
        .mean()
        .reset_index()
        .rename(columns={
            "native_mean": "native_group_mean",
            "nonnative_mean": "nonnative_group_mean",
        })
    )
    df = df.merge(group_means, on=["domain", "type"], how="left")
    return df


###############################################################
SECTIONS = {
    "ADP":              ["ADP_SH1", "ADP_S1", "ADP_Ploop", "ADP_Purine"],
    "Ploop-Phosphate":  ["Ploop_Pi@P", "Ploop_ADP@PB"],
    "Switch1":          ["S1_SH1", "S1_U50", "S1_L50", "S1_Ohelix", "S1_Relay"],
    "Cleft":            ["S1_S2", "U50_L50", "HLH_L2", "L3_L2", "Ohelix_Relay"],
}


def split_into_section_dataframes(sections):
    parent = Path(PARENT_FOLDER)
    df = process_native_files(parent / "results_wt", parent / "results_d239n")

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


###############################################################
def _add_stars(ax, x_pos, y_top, stars):
    """Draw a star annotation centred above a bar pair."""
    if stars:
        ax.text(x_pos, y_top * 1.08, stars, ha="center", va="bottom", fontsize=13)


###############################################################
def plot_native_section(section_name, section_df):
    domains = section_df["domain"].unique()
    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    wt_means, mut_means, wt_sems, mut_sems = [], [], [], []
    stars_list = []
    stats_dict = {}

    for domain in domains:
        wt_vals  = section_df[(section_df["domain"] == domain) & (section_df["type"] == "wt")]["native_mean"].astype(float)
        mut_vals = section_df[(section_df["domain"] == domain) & (section_df["type"] == "mut")]["native_mean"].astype(float)

        wt_means.append(wt_vals.mean())
        mut_means.append(mut_vals.mean())
        wt_sems.append(wt_vals.sem())
        mut_sems.append(mut_vals.sem())
        stats_dict[domain] = (wt_vals, mut_vals)

        if len(wt_vals) == 3 and len(mut_vals) == 3:
            p = ttest_ind(wt_vals, mut_vals).pvalue
        else:
            p = np.nan
        stars_list.append(pval_to_stars(p) if not np.isnan(p) else "")

    ax.bar(x - width/2, wt_means,  width, yerr=wt_sems,  label="WT",  color="skyblue", capsize=5)
    ax.bar(x + width/2, mut_means, width, yerr=mut_sems, label="Mut", color="salmon",  capsize=5)

    for i, domain in enumerate(domains):
        wt_vals, mut_vals = stats_dict[domain]
        ax.scatter(np.repeat(x[i] - width/2, len(wt_vals)),  wt_vals,  color="blue", s=40)
        ax.scatter(np.repeat(x[i] + width/2, len(mut_vals)), mut_vals, color="red",  s=40)

    for i, stars in enumerate(stars_list):
        y_top = max(wt_means[i] + wt_sems[i], mut_means[i] + mut_sems[i])
        _add_stars(ax, x[i], y_top, stars)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.set_ylabel("Mean Native Contacts")
    ax.set_title(f"{section_name} — Native Contacts (WT vs D239N)")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, f"{section_name}_native.png")
    plt.show()


###############################################################
def plot_nonnative_section(section_name, section_df):
    domains = section_df["domain"].unique()
    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    wt_means, mut_means, wt_sems, mut_sems = [], [], [], []
    stars_list = []
    stats_dict = {}

    for domain in domains:
        wt_vals  = section_df[(section_df["domain"] == domain) & (section_df["type"] == "wt")]["nonnative_mean"].astype(float)
        mut_vals = section_df[(section_df["domain"] == domain) & (section_df["type"] == "mut")]["nonnative_mean"].astype(float)

        wt_means.append(wt_vals.mean())
        mut_means.append(mut_vals.mean())
        wt_sems.append(wt_vals.sem())
        mut_sems.append(mut_vals.sem())
        stats_dict[domain] = (wt_vals, mut_vals)

        if len(wt_vals) == 3 and len(mut_vals) == 3:
            p = ttest_ind(wt_vals, mut_vals).pvalue
        else:
            p = np.nan
        stars_list.append(pval_to_stars(p) if not np.isnan(p) else "")

    ax.bar(x - width/2, wt_means,  width, yerr=wt_sems,  label="WT",  color="skyblue", capsize=5)
    ax.bar(x + width/2, mut_means, width, yerr=mut_sems, label="Mut", color="salmon",  capsize=5)

    for i, domain in enumerate(domains):
        wt_vals, mut_vals = stats_dict[domain]
        ax.scatter(np.repeat(x[i] - width/2, len(wt_vals)),  wt_vals,  color="blue", s=40)
        ax.scatter(np.repeat(x[i] + width/2, len(mut_vals)), mut_vals, color="red",  s=40)

    for i, stars in enumerate(stars_list):
        y_top = max(wt_means[i] + wt_sems[i], mut_means[i] + mut_sems[i])
        _add_stars(ax, x[i], y_top, stars)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.set_ylabel("Mean Non-Native Contacts")
    ax.set_title(f"{section_name} — Non-Native Contacts (WT vs D239N)")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, f"{section_name}_nonnative.png")
    plt.show()


###############################################################
def plot_native_nonnative_axis(section_name, section_df):
    all_domains = section_df["domain"].unique()

    filtered_domains = []
    for domain in all_domains:
        sub = section_df[section_df["domain"] == domain]
        wt_nat  = sub[sub["type"] == "wt"]["native_mean"].astype(float)
        mut_nat = sub[sub["type"] == "mut"]["native_mean"].astype(float)
        wt_non  = sub[sub["type"] == "wt"]["nonnative_mean"].astype(float)
        mut_non = sub[sub["type"] == "mut"]["nonnative_mean"].astype(float)

        if wt_nat.sum() == 0 and mut_nat.sum() == 0 and wt_non.sum() == 0 and mut_non.sum() == 0:
            print(f"  Skipping domain {domain} — all values are zero.")
            continue
        filtered_domains.append(domain)

    if not filtered_domains:
        print(f"Skipping section '{section_name}' — all domains are zero.")
        return

    domains = np.array(filtered_domains)
    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    wt_nat_means,  mut_nat_means  = [], []
    wt_nat_sems,   mut_nat_sems   = [], []
    wt_non_means,  mut_non_means  = [], []
    wt_non_sems,   mut_non_sems   = [], []
    stars_native, stars_nonnative = [], []
    stats_native, stats_nonnative = {}, {}

    for domain in domains:
        sub = section_df[section_df["domain"] == domain]
        wt_nat  = sub[sub["type"] == "wt"]["native_mean"].astype(float)
        mut_nat = sub[sub["type"] == "mut"]["native_mean"].astype(float)
        wt_non  = sub[sub["type"] == "wt"]["nonnative_mean"].astype(float)
        mut_non = sub[sub["type"] == "mut"]["nonnative_mean"].astype(float)

        stats_native[domain]    = (wt_nat,  mut_nat)
        stats_nonnative[domain] = (wt_non,  mut_non)

        wt_nat_means.append(wt_nat.mean())
        mut_nat_means.append(mut_nat.mean())
        wt_nat_sems.append(wt_nat.sem())
        mut_nat_sems.append(mut_nat.sem())

        wt_non_means.append(-wt_non.mean())
        mut_non_means.append(-mut_non.mean())
        wt_non_sems.append(wt_non.sem())
        mut_non_sems.append(mut_non.sem())

        def _stars(a, b):
            if len(a) == 3 and len(b) == 3:
                return pval_to_stars(ttest_ind(a, b).pvalue)
            return ""

        stars_native.append(_stars(wt_nat, mut_nat))
        stars_nonnative.append(_stars(wt_non, mut_non))

    ax.bar(x - width/2, wt_nat_means,  width, yerr=wt_nat_sems,  label="WT Native",     color="skyblue", capsize=5)
    ax.bar(x + width/2, mut_nat_means, width, yerr=mut_nat_sems, label="Mut Native",    color="salmon",  capsize=5)
    ax.bar(x - width/2, wt_non_means,  width, yerr=wt_non_sems,  label="WT Non-native", color="blue",    alpha=0.4, capsize=5)
    ax.bar(x + width/2, mut_non_means, width, yerr=mut_non_sems, label="Mut Non-native",color="red",     alpha=0.4, capsize=5)

    for i, domain in enumerate(domains):
        wt_nat,  mut_nat  = stats_native[domain]
        wt_non,  mut_non  = stats_nonnative[domain]
        ax.scatter(np.repeat(x[i] - width/2, len(wt_nat)),   wt_nat,  color="blue", s=40)
        ax.scatter(np.repeat(x[i] + width/2, len(mut_nat)),  mut_nat, color="red",  s=40)
        ax.scatter(np.repeat(x[i] - width/2, len(wt_non)),  -wt_non,  color="blue", s=40, alpha=0.6)
        ax.scatter(np.repeat(x[i] + width/2, len(mut_non)), -mut_non, color="red",  s=40, alpha=0.6)

    for i, (sn, sp) in enumerate(zip(stars_native, stars_nonnative)):
        if sn:
            y_top = max(wt_nat_means[i] + wt_nat_sems[i], mut_nat_means[i] + mut_nat_sems[i])
            _add_stars(ax, x[i], y_top, sn)
        if sp:
            y_bot = min(wt_non_means[i] - wt_non_sems[i], mut_non_means[i] - mut_non_sems[i])
            ax.text(x[i], y_bot * 1.08, sp, ha="center", va="top", fontsize=13)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.set_ylabel("Contacts (Native ↑, Non-native ↓)")
    ax.set_title(f"{section_name} — Native vs Non-native Contacts (WT vs D239N)")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, f"{section_name}_native_nonnative_axis.png")
    plt.show()


###############################################################
# Main
section_dfs = split_into_section_dataframes(SECTIONS)
for section_name, section_df in section_dfs.items():
    plot_native_section(section_name, section_df)
    plot_nonnative_section(section_name, section_df)
    plot_native_nonnative_axis(section_name, section_df)