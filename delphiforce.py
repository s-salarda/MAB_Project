# ================================================================
# Delphiforce data analysis
# ================================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import seaborn as sns
from scipy import stats

# ================================================================
# File paths
# ================================================================
FILE_DIR = r"C:\Users\salar\MAB_project\Delphi\MAB_DelphiForce_4\files_pqr"
OUT_CSV  = r"C:\Users\salar\MAB_project\Delphi\MAB_DelphiForce_4\files_csv\all_results_py.csv"
FIGURES_DIR     = r"C:\Users\salar\MAB_project\Delphi\figures"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================================================================
# Global parameters
# ================================================================
genotype_colors = {"WT": "black", "D239N": "green", "K637E": "red"}
domain_colors   = ["#AED6F1","#A9DFBF","#F9E79F","#F5CBA7","#D2B4DE",
                   "#F1948A","#76D7C4","#F0B27A","#85C1E9","#82E0AA"]
GENOTYPE_ORDER  = ["WT", "D239N", "K637E"]

DOMAIN_LABELS = {
    "Upper 50kDa Domain": [(215,231), (266,453), (604,621)],
    "Lower 50kDa Domain": [(472,566), (578,590), (645,665)],
    "S1":                 [(234,244)],
    "SH1":                [(672,685)],
    "Purine Loop":        [(126,131)],
    "P Loop":             [(179,183)],
    "N Terminal":         [(113,123), (666,671), (170,175), (454,460), (244,265)],
    "Loop 2":             [(624,646)],
    "Loop 3":             [(567,577)],
    "S2":                 [(462,472)],
}

# ================================================================
# Read all data and save to CSV
# ================================================================
GENOTYPE_MAP = {"wt": "WT", "k637e": "K637E", "d239n": "D239N"}  # Map raw genotype strings to standardized labels

COL_SPECS = [(0,3),(4,5),(6,10),(18,25),(28,35),(38,45),(48,55),(58,65)] # fixed-width column specs based on sample file
COL_NAMES = ["Residue","Chain","ID","Net_Charge","G","Fx","Fy","Fz"] # column names for DataFrame

# Note: The following code block is commented out because it only needs to be run once to create the CSV
# filelist = [f for f in os.listdir(FILE_DIR) if f.endswith(".residue")]
# print(f"Found {len(filelist)} files")
# all_data_df = []
# for file_name in filelist:
#     geno_raw, sim, frame = re.match(r"([^_]+)_sim(\d+)_frame(\d+)", file_name).groups()
#     df = pd.read_fwf(
#         os.path.join(FILE_DIR, file_name),
#         colspecs   = COL_SPECS,
#         names      = COL_NAMES,
#         skiprows   = 1,
#         skipfooter = 2,
#         engine     = "python"
#     )
#     df["Residue_ID"] = df["ID"].astype(int).astype(str) + "_" + df["Residue"]
#     df["Genotype"]   = GENOTYPE_MAP.get(geno_raw.lower(), "ERROR")
#     df["sim"]        = int(sim)
#     df["frame"]      = int(frame)
#     df["file_name"]  = file_name
#     all_data_df.append(df)
# df = pd.concat(all_data_df, ignore_index=True)
# df["Genotype"] = pd.Categorical(df["Genotype"], categories=GENOTYPE_ORDER, ordered=True)
# df = df.sort_values("frame").reset_index(drop=True)
# df.to_csv(OUT_CSV, index=False)
# print(f"Saved → {OUT_CSV}")

# ================================================================
# Load CSV into DataFrame
# ================================================================
df = pd.read_csv(OUT_CSV, low_memory=False)
df["Genotype"] = pd.Categorical(df["Genotype"], categories=GENOTYPE_ORDER, ordered=True) # Ensure Genotype is categorical with correct order

for col in ["ID", "Net_Charge", "G", "Fx", "Fy", "Fz"]: # Convert to numeric, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors="coerce")

n_before = len(df)
df = df.dropna(subset=["ID", "G"])
df["ID"] = df["ID"].astype(int) # Ensure ID is integer after dropping NaNs (some rows may have had non-numeric IDs that became NaN)
print(f"Dropped {n_before - len(df):,} malformed rows ({len(df):,} remaining)")

# ================================================================
# Sanity check
# ================================================================
n_files       = len([f for f in os.listdir(FILE_DIR) if f.endswith(".residue")])
rows_per_file = df.groupby("file_name").size()
expected      = int(rows_per_file.mode()[0])
bad_files     = rows_per_file[rows_per_file != expected]

print(f"Total files read:        {df['file_name'].nunique()}")
print(f"Expected rows per file:  {expected}")
print(f"Total rows:              {len(df):,}")
print(f"Files with wrong count:  {len(bad_files)} (likely incomplete/missing data)")

# ================================================================
# Convert binding energy to force & average per simulation
# ================================================================
df["binding_force"] = df["G"] * -41.14   # convert kT/Å to nN, flip sign

df_avg_sim = (df.groupby(["Residue_ID", "ID", "Genotype", "sim"], observed=True) # average by simulation to reduce noise before ANOVA
                .agg(binding_force=("binding_force", "mean"))
                .reset_index())

# ================================================================
# Filter using G threshold (like R script)
# ================================================================

# G threshold filtering - match R script exactly
df_threshold = df_avg_sim.groupby("Residue_ID").agg({"binding_force": ["min", "max"]}).reset_index()
df_threshold.columns = ["Residue_ID", "binding_force_min", "binding_force_max"]

# Convert back to G values for threshold comparison (reverse the *-41.14 conversion)
df_threshold["G_min"] = df_threshold["binding_force_min"] / -41.14
df_threshold["G_max"] = df_threshold["binding_force_max"] / -41.14

important_residues_g = set(df_threshold[(df_threshold["G_min"] < -0.01) | 
                                        (df_threshold["G_max"] > 0.01)]["Residue_ID"])

df_important_g = df_avg_sim[df_avg_sim["Residue_ID"].isin(important_residues_g)].copy()
print(f"Important residues (G threshold): {len(important_residues_g)}")

# Average by genotype for overview plots
df_important_avgGt_g = (df_important_g.groupby(["Residue_ID", "ID", "Genotype"], observed=True)
                                      .agg({"binding_force": "mean"})
                                      .reset_index())

# ================================================================
# ANOVA calculations for star annotations
# ================================================================
def calculate_anova_pvalues(df_avg_sim):
    """Calculate ANOVA p-values for each residue across genotypes"""
    print("Calculating ANOVA p-values for star annotations...")
    anova_rows = []
    for (res_id, id_val), grp in df_avg_sim.groupby(["Residue_ID", "ID"]):
        groups = [g["binding_force"].values for _, g in grp.groupby("Genotype", observed=True)]
        try:
            _, p = stats.f_oneway(*groups)
        except Exception:
            p = np.nan
        anova_rows.append({"Residue_ID": res_id, "ID": id_val, "p_value": p})
    
    return pd.DataFrame(anova_rows)

def pvalue_to_stars(p):
    """Convert p-value to significance stars"""
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""  # Don't show "ns"

# Calculate ANOVA for all residues
anova_all = calculate_anova_pvalues(df_avg_sim)

# ================================================================
# Plot Functions
# ================================================================
def plot_domain(domain_name, domain_ranges):
    """
    Plot binding force for ALL important residues (G threshold) in a domain.
    ANOVA stars appear above bars only if significant.
    Uses global: df_avg_sim, anova_all, important_residues_g
    """
    
    # Filter to residues in this domain that passed G threshold
    in_domain = df_important_g[
        df_important_g["ID"].apply(lambda x: any(lo <= x <= hi for lo, hi in domain_ranges))
    ].copy()

    if in_domain.empty:
        print(f"  No important residues in {domain_name} — skipping")
        return

    order = in_domain.drop_duplicates("Residue_ID").sort_values("ID")["Residue_ID"].tolist()
    n = len(order) # number of unique residues in this domain that passed G threshold

    # Figure width scales with number of residues, but capped to prevent extreme sizes
    fig_w = min(49, max(6, n * 0.55))
    fig, ax = plt.subplots(figsize=(fig_w, 10))

    # Track max heights for star placement
    max_heights = {}

    # Segmented bars
    x = np.arange(n) * 1.2
    bar_w = 0.25

    # Plot bars for each genotype
    for j, geno in enumerate(GENOTYPE_ORDER):
        subset = in_domain[in_domain["Genotype"] == geno].groupby("Residue_ID")["binding_force"].mean()
        vals   = [subset.loc[r] if r in subset.index else 0.0 for r in order]
        
        # Track max heights for star placement
        for i, (res, val) in enumerate(zip(order, vals)):
            if res not in max_heights:
                max_heights[res] = abs(val)
            else:
                max_heights[res] = max(max_heights[res], abs(val))
        
        offset = (j - 1) * bar_w
        ax.bar(x + offset, vals, width=bar_w,
               color=genotype_colors[geno], label=geno, zorder=2,
               edgecolor="white", linewidth=0.5)

    # Add ANOVA significance stars (only if p < 0.05, otherwise blank)
    for i, res in enumerate(order):
        if res in anova_all["Residue_ID"].values:
            p_val = anova_all[anova_all["Residue_ID"] == res]["p_value"].iloc[0]
            stars = pvalue_to_stars(p_val)
            if stars:  # Only show if significant (*, **, ***)
                star_y = max_heights[res] + 0.3 if res in max_heights else 5.5
                ax.text(i * 1.2, star_y, stars, ha="center", va="bottom", 
                       fontsize=10, fontweight="bold", color="black")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-6, 7)

    # Add dotted vertical lines between residues
    for i in range(1, n):
        ax.axvline(x[i] - 0.6, color="gray", linestyle=":", alpha=0.7, zorder=1)

    # Add background colors for ALL domains
    if len(domain_ranges) == 1:
        # Single range domain - one background
        ax.axvspan(-0.6, (n-1) * 1.2 + 0.6, color=domain_colors[0], alpha=0.3, zorder=0)
        ax.text(0.1, 6.2, domain_name, fontsize=9, fontweight="bold", color="black", va="top")
    else:
        # Multi-range domain - color each range separately
        pos_map = {res: i * 1.2 for i, res in enumerate(order)}
        for i, (lo, hi) in enumerate(domain_ranges):
            in_band = [r for r in order
                       if lo <= in_domain[in_domain["Residue_ID"]==r]["ID"].iloc[0] <= hi]
            if not in_band:
                continue
            positions = [pos_map[r] for r in in_band]
            xmin, xmax = min(positions) - 0.6, max(positions) + 0.6
            ax.axvspan(xmin, xmax, color=domain_colors[i % len(domain_colors)], 
                      alpha=0.3, zorder=0)
            ax.text(xmin + 0.1, 6.2, domain_name,
                    fontsize=8, fontweight="bold", color="black", va="top")
    
    # Set x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right",
                       fontsize=max(5, min(10, 120 // n)))
    ax.set_xlabel("Residue", fontweight="bold")
    ax.set_ylabel("Binding Force (nN)", fontweight="bold")
    
    ax.set_title(f"{domain_name}", fontweight="bold", pad=30)
    
    # Legend under the title 
    legend_elements = [
        patch.Patch(facecolor='none', edgecolor='none', label='Genotype:'),  # Title as first item
        patch.Patch(facecolor='black', edgecolor='black', label='WT'),
        patch.Patch(facecolor='green', edgecolor='black', label='D239N'),
        patch.Patch(facecolor='red', edgecolor='black', label='K637E')
    ]
    ax.legend(handles=legend_elements, loc='upper center', ncol=4, frameon=False,
              bbox_to_anchor=(0.5, 1.05), handlelength=1.5, handleheight=1.5,
              columnspacing=1)
    
    # Add star legend
    ax.text(0.98, 0.99, "ANOVA: * p<0.05, ** p<0.01, *** p<0.001", 
            transform=ax.transAxes, fontsize=10, 
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

    sns.despine(ax=ax)
    plt.tight_layout()

    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", domain_name)
    save_path = os.path.join(FIGURES_DIR, f"{safe_name}_Gthreshold_with_ANOVA.png")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_all_significant():
    """
    Overview bar plot of important residues (G threshold) with ANOVA significance stars.
    Organized by domain, then by residue order within domain.
    Uses global: df_important_avgGt_g, anova_all, DOMAIN_LABELS
    """

    if df_important_avgGt_g.empty:
        print("No important residues to plot.")
        return

    # Organize residues by domain first, then by ID within domain
    domain_residues = []
    for domain_name, domain_ranges in DOMAIN_LABELS.items():
        domain_res = []
        for lo, hi in domain_ranges:
            res_in_range = df_important_avgGt_g[(df_important_avgGt_g["ID"] >= lo) & 
                                                (df_important_avgGt_g["ID"] <= hi)]
            if not res_in_range.empty:
                sorted_res = res_in_range.drop_duplicates("Residue_ID").sort_values("ID")["Residue_ID"].tolist()
                domain_res.extend(sorted_res)
        if domain_res:
            domain_residues.append((domain_name, domain_res))

    # Create ordered list: domain by domain, residues sorted within each
    order = []
    domain_boundaries = []  # Track where each domain starts/ends for background coloring
    current_pos = 0
    
    for domain_name, res_list in domain_residues:
        order.extend(res_list)
        domain_boundaries.append((domain_name, current_pos, current_pos + len(res_list) - 1))
        current_pos += len(res_list)

    n = len(order)
    if n == 0:
        print("No important residues found in any domain")
        return

    fig_w = min(49, max(17.5, n * 0.4))
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    # Track max heights for star placement
    max_heights = {}

    # Segmented bars (wider spacing between residues)
    x = np.arange(n) * 1.2  # More spacing between residue groups
    bar_w = 0.25            # Thinner individual bars

    for j, geno in enumerate(GENOTYPE_ORDER):
        subset = df_important_avgGt_g[df_important_avgGt_g["Genotype"] == geno].set_index("Residue_ID")["binding_force"]
        vals   = [subset.loc[r] if r in subset.index else 0.0 for r in order]
        
        # Track max heights for star placement
        for i, (res, val) in enumerate(zip(order, vals)):
            if res not in max_heights:
                max_heights[res] = abs(val)
            else:
                max_heights[res] = max(max_heights[res], abs(val))
        
        offset = (j - 1) * bar_w  # Center around 0
        ax.bar(x + offset, vals, width=bar_w,
               color=genotype_colors[geno], label=geno, zorder=2, 
               edgecolor="white", linewidth=0.5)

    # Add ANOVA significance stars
    for i, res in enumerate(order):
        if res in anova_all["Residue_ID"].values:
            p_val = anova_all[anova_all["Residue_ID"] == res]["p_value"].iloc[0]
            stars = pvalue_to_stars(p_val)
            if stars:  # Only show stars, not "ns"
                star_y = max_heights[res] + 0.3 if res in max_heights else 5.5
                ax.text(i * 1.2, star_y, stars, ha="center", va="bottom", 
                       fontsize=10, fontweight="bold", color="black")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-6, 7)  # Extra space for stars

    # Add dotted vertical lines between residues
    for i in range(1, n):
        ax.axvline(x[i] - 0.6, color="gray", linestyle=":", alpha=0.7, zorder=1)

    # Add domain background colors and labels
    for i, (domain_name, start_idx, end_idx) in enumerate(domain_boundaries):
        if start_idx <= end_idx:  # Valid range
            xmin = start_idx * 1.2 - 0.6
            xmax = end_idx * 1.2 + 0.6
            
            color_idx = i % len(domain_colors)
            ax.axvspan(xmin, xmax, color=domain_colors[color_idx], alpha=0.3, zorder=0)
            
            # Add domain label in top-left corner
            ax.text(xmin + 0.1, 6.2, domain_name,
                    fontsize=9, fontweight="bold", color="black",
                    verticalalignment="top")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right",
                       fontsize=max(5, min(9, 120 // n)))
    ax.set_xlabel("Residue", fontweight="bold")
    ax.set_ylabel("Binding Force (nN)", fontweight="bold")
    
    # Title
    ax.set_title("Important Residues over Domains", fontweight="bold", pad=30)
    
    # Create inline legend with Genotype: label
    legend_elements = [
        patch.Patch(facecolor='none', edgecolor='none', label='Genotype:'),
        patch.Patch(facecolor='black', edgecolor='white', label='WT'),
        patch.Patch(facecolor='green', edgecolor='white', label='D239N'),
        patch.Patch(facecolor='red', edgecolor='white', label='K637E')
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
              ncol=4, frameon=False, handlelength=1.5, handleheight=1.5,
              )

    # Add star legend in top-right corner
    ax.text(0.98, 0.99, "ANOVA: * p<0.05, ** p<0.01, *** p<0.001", 
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

    sns.despine(ax=ax)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "Important_Residues_G_threshold_with_ANOVA.png")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_pct_change():
    """
    Bar plot of % change in binding force from WT (organized by domain) with ANOVA stars.
    Uses global: df_important_avgGt_g, anova_all, DOMAIN_LABELS
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if df_important_avgGt_g.empty:
        print("No important residues to plot.")
        return

    # Calculate percent change from WT
    wt_ref = df_important_avgGt_g[df_important_avgGt_g["Genotype"] == "WT"].set_index("ID")["binding_force"].rename("binding_force_wt")
    df_pct = df_important_avgGt_g.merge(wt_ref.reset_index(), on="ID", how="left")
    df_pct["pct_change"] = (df_pct["binding_force"] - df_pct["binding_force_wt"]) / df_pct["binding_force_wt"] * 100
    df_pct = df_pct[df_pct["Genotype"] != "WT"]

    # Organize residues by domain first, then by ID within domain
    domain_residues = []
    for domain_name, domain_ranges in DOMAIN_LABELS.items():
        domain_res = []
        for lo, hi in domain_ranges:
            res_in_range = df_pct[(df_pct["ID"] >= lo) & (df_pct["ID"] <= hi)]
            if not res_in_range.empty:
                sorted_res = res_in_range.drop_duplicates("Residue_ID").sort_values("ID")["Residue_ID"].tolist()
                domain_res.extend(sorted_res)
        if domain_res:
            domain_residues.append((domain_name, domain_res))

    # Create ordered list: domain by domain
    order = []
    domain_boundaries = []
    current_pos = 0
    
    for domain_name, res_list in domain_residues:
        order.extend(res_list)
        domain_boundaries.append((domain_name, current_pos, current_pos + len(res_list) - 1))
        current_pos += len(res_list)

    n = len(order)
    if n == 0:
        print("No important residues found for percent change")
        return

    fig_w = min(49, max(17.5, n * 0.35))
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    mut_genotypes = ["D239N", "K637E"]
    x = np.arange(n) * 1.2  # CHANGED: Added 1.2 spacing
    bar_w = 0.35

    # Track max heights for star placement
    max_heights = {}

    for j, geno in enumerate(mut_genotypes):
        subset = df_pct[df_pct["Genotype"] == geno].set_index("Residue_ID")["pct_change"]
        vals   = [subset.loc[r] if r in subset.index else 0.0 for r in order]
        
        # Track max heights for stars
        for i, (res, val) in enumerate(zip(order, vals)):
            if res not in max_heights:
                max_heights[res] = abs(val)
            else:
                max_heights[res] = max(max_heights[res], abs(val))
        
        offset = (j - 0.5) * bar_w
        ax.bar(x + offset, vals, width=bar_w,
               color=genotype_colors[geno], label=geno, zorder=2,
               edgecolor="white", linewidth=0.5)

    # Add ANOVA significance stars  
    for i, res in enumerate(order):
        if res in anova_all["Residue_ID"].values:
            p_val = anova_all[anova_all["Residue_ID"] == res]["p_value"].iloc[0]
            stars = pvalue_to_stars(p_val)
            if stars:
                star_y = max_heights[res] + 5 if res in max_heights else 15
                ax.text(i * 1.2, star_y, stars, ha="center", va="bottom",  # CHANGED: i * 1.2
                       fontsize=10, fontweight="bold", color="black")

    ax.axhline(0, color="black", linewidth=0.8)
    
    # Dotted vertical lines between residues
    for i in range(1, n):
        ax.axvline(x[i] - 0.6, color="gray", linestyle=":", alpha=0.7, zorder=1)

    # Add domain background colors
    for i, (domain_name, start_idx, end_idx) in enumerate(domain_boundaries):
        if start_idx <= end_idx:
            xmin = start_idx * 1.2 - 0.6  # CHANGED: Added 1.2 spacing
            xmax = end_idx * 1.2 + 0.6    # CHANGED: Added 1.2 spacing
            
            color_idx = i % len(domain_colors)
            ax.axvspan(xmin, xmax, color=domain_colors[color_idx], alpha=0.3, zorder=0)
            
            # Get y-range for label positioning
            y_min, y_max = ax.get_ylim()
            label_y = y_max - (y_max - y_min) * 0.1
            
            # Add domain label
            ax.text(xmin + 0.1, label_y, domain_name,
                    fontsize=9, fontweight="bold", color="black", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right",
                       fontsize=max(5, min(10, 120 // n)))
    ax.set_xlabel("Residue", fontweight="bold")
    ax.set_ylabel("% Change from WT", fontweight="bold")
    
    # Title
    ax.set_title("Important Residues: % Change from WT", fontweight="bold", pad=30)
    
    # Create inline legend with Genotype: label
    legend_elements = [
        patch.Patch(facecolor='none', edgecolor='none', label='Genotype:'),
        patch.Patch(facecolor='green', edgecolor='white', label='D239N'),
        patch.Patch(facecolor='red', edgecolor='white', label='K637E')
    ]
    
    leg = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
                    ncol=3, frameon=False, handlelength=1.5, handleheight=1.5)
    
    # Make legend text bold
    for text in leg.get_texts():
        text.set_fontweight('bold')
    
    # Add star legend in top-right corner
    ax.text(0.98, 0.99, "ANOVA: * p<0.05, ** p<0.01, *** p<0.001", 
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    sns.despine(ax=ax)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "Important_Residues_PctChange_with_ANOVA.png")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

# ================================================================
# Main Execute
# ================================================================

# Per-domain plots
print("\nPlotting per-domain figures...")
for domain_name, domain_ranges in DOMAIN_LABELS.items():
    plot_domain(domain_name, domain_ranges)

# Overview plots using G threshold + ANOVA stars
print("\nPlotting overview figures with G threshold + ANOVA stars...")
plot_all_significant()
plot_pct_change()
