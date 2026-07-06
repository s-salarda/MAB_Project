# Distance Analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from scipy import stats

file_path = r"C:\Users\salar\Desktop\xbc_pps_simulations\Distances"
excel = r"C:\Users\salar\Desktop\xbc_pps_simulations\Distances\distances.xlsx"
####################################################################################################################
HEADER_RE = re.compile(
    r"(#?\d+\s+)?\w+\s+\d+\.\w+\s+\w+\s+<->\s+(#?\d+\s+)?\w+\s+\d+\.\w+\s+\w+"
)

def parse_reaction_file(file_path, max_time=500, skip_pairs=None):
    averages = {}
    current_title = None
    current_rows = []

    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.lower().startswith("distance"):
                continue

            # Reaction header
            if HEADER_RE.match(line):
                print("Found section header:", line)
                # flush previous block
                if current_title and current_rows:
                    df = pd.DataFrame(current_rows, columns=["Time", "Distance"])
                    df = df[df["Time"] <= max_time]
                    if not df.empty:
                        averages[current_title] = df["Distance"].mean()
                    

                # Check if this header should be skipped
                if skip_pairs:
                    skip_this = False
                    for kw1, kw2 in skip_pairs:
                        if kw1.lower() in line.lower() and kw2.lower() in line.lower():
                            skip_this = True
                            break
                    if skip_this:
                        current_title = None
                        current_rows = []
                        continue

                # Otherwise keep it
                current_title = line
                current_rows = []
                continue

            # Data row
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if len(nums) >= 2:
                try:
                    time = float(nums[0])
                    dist = float(nums[1])
                    if current_title:
                        current_rows.append([time, dist])
                except ValueError:
                    pass

    # flush last block
    if current_title and current_rows:
        df = pd.DataFrame(current_rows, columns=["Time", "Distance"])
        df = df[df["Time"] <= max_time]
        if not df.empty:
            averages[current_title] = df["Distance"].mean()

    return averages


def reactions_summary_to_excel(parent_folder, keyword, excel_out= excel, max_time=500):
    """
    Walk through folder, find .txt files with keyword in name.
    Parse each file, compute averages per reaction.
    Write summary Excel with:
      - One sheet for all WT files
      - One sheet for all MUT files
    """
    wt_rows, mut_rows = [], []

    for root, _, filenames in os.walk(parent_folder):
        for f in filenames:
            if f.endswith(".txt") and keyword in f:
                file_path = os.path.join(root, f)
                averages = parse_reaction_file(file_path, max_time=max_time)
                for title, avg_dist in averages.items():
                    row = {
                        "Reaction": title,
                        "File": f,
                        "Avg_Distance": avg_dist
                    }
                    if "wt" in f.lower():
                        wt_rows.append(row)
                    elif "mut" in f.lower():
                        mut_rows.append(row)

    with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
        if wt_rows:
            pd.DataFrame(wt_rows).to_excel(writer, sheet_name="WT", index=False)
        if mut_rows:
            pd.DataFrame(mut_rows).to_excel(writer, sheet_name="MUT", index=False)
    return wt_rows, mut_rows

def average_per_file_with_keywords(parent_folder, keywords, max_time=500, skip_pairs=None):
    results = {}
    for root, _, filenames in os.walk(parent_folder):
        for f in filenames:
            # Debug: show every file being scanned
            print("Scanning file:", f)

            # Check if it matches your keywords
            if f.endswith(".txt") and all(k.lower() in f.lower() for k in keywords):
                print("Matched file:", f)  # Debug: confirm match

                file_path = os.path.join(root, f)
                # pass skip_pairs into parse_reaction_file
                averages = parse_reaction_file(file_path, max_time=max_time, skip_pairs=skip_pairs)
                if averages:
                    overall_avg = sum(averages.values()) / len(averages)
                    results[f] = overall_avg

    # Print results inside the function
    for file, avg in results.items():
        print(file, ":", avg)

    return results

def format_p(p_val):
    """
    Format p-values in a clean journal style:
    - p < 0.001 if very small
    - otherwise rounded to 3 decimals
    """
    if p_val < 0.05:
        return f"p = {p_val:.4f}"
    else:
        return f" "

def plot_wt_mut_comparison(wt_dict, mut_dict, custom_p=None, title="WT vs MUT Comparison"):
    """
    Takes WT and MUT dicts {filename: avg_distance}, creates a bar plot with SEM,
    overlays individual replicate values, and performs a t-test.
    Title includes both calculated and optional custom p-value.
    """
    # Convert dict values to arrays
    wt_values = np.array(list(wt_dict.values()))
    mut_values = np.array(list(mut_dict.values()))

    # Compute means and SEMs
    wt_mean, wt_sem = wt_values.mean(), stats.sem(wt_values)
    mut_mean, mut_sem = mut_values.mean(), stats.sem(mut_values)

    # T-test
    t_stat, p_val = stats.ttest_ind(wt_values, mut_values, equal_var=False)

    # Format p-values nicely
    calc_str = format_p(p_val)
    if custom_p is not None:
        custom_str = format_p(custom_p)
        full_title = f"{title} ({calc_str}, custom {custom_str})"
    else:
        full_title = f"{title} ({calc_str})"

    # Bar plot
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    ax.bar([0,1], [wt_mean, mut_mean],
           yerr=[wt_sem, mut_sem],
           capsize=5, color=["skyblue","salmon"],
           tick_label=["WT","MUT"])
    
    # Overlay individual points
    ax.scatter(np.zeros_like(wt_values), wt_values, color="blue", zorder=10)
    ax.scatter(np.ones_like(mut_values), mut_values, color="red", zorder=10)

    ax.set_ylabel("Average Distance")
    ax.set_title(full_title)

    plt.tight_layout()
    plt.show()

    return t_stat, p_val

def plot_combined_wt_mut_comparisons(wt_dicts, mut_dicts, residue_labels, title="WT vs MUT for Residue Ranges"):
    """
    Combines multiple WT vs MUT comparisons into one grouped bar plot.
    
    Parameters:
    - wt_dicts: list of dicts {filename: avg_distance} for each residue range
    - mut_dicts: list of dicts {filename: avg_distance} for each residue range
    - residue_labels: list of strings like "239-320", "239-523", etc.
    - title: plot title
    """
    assert len(wt_dicts) == len(mut_dicts) == len(residue_labels), "Input lists must be same length"

    wt_means, wt_sems = [], []
    mut_means, mut_sems = [], []
    p_values = []

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for i, (wt_dict, mut_dict) in enumerate(zip(wt_dicts, mut_dicts)):
        wt_vals = np.array(list(wt_dict.values()))
        mut_vals = np.array(list(mut_dict.values()))

        wt_mean, wt_sem = wt_vals.mean(), stats.sem(wt_vals)
        mut_mean, mut_sem = mut_vals.mean(), stats.sem(mut_vals)

        wt_means.append(wt_mean)
        wt_sems.append(wt_sem)
        mut_means.append(mut_mean)
        mut_sems.append(mut_sem)

        t_stat, p_val = stats.ttest_ind(wt_vals, mut_vals, equal_var=False)
        p_values.append(p_val)

        # Overlay individual points
        ax.scatter([i - 0.15] * len(wt_vals), wt_vals, color="blue", zorder=10)
        ax.scatter([i + 0.15] * len(mut_vals), mut_vals, color="red", zorder=10)

    # Bar plot
    x = np.arange(len(residue_labels))
    ax.bar(x - 0.15, wt_means, yerr=wt_sems, capsize=5, color="skyblue", width=0.3, label="WT")
    ax.bar(x + 0.15, mut_means, yerr=mut_sems, capsize=5, color="salmon", width=0.3, label="MUT")

    # Annotate p-values
    for i, p_val in enumerate(p_values):
        offset = max(wt_sems[i], mut_sems[i]) + 1.0  # add SEM + extra padding
        ax.text(x[i], max(wt_means[i], mut_means[i]) + offset, format_p(p_val),
                ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1.5 * max(wt_means + mut_means))
    ax.set_xticks(x)
    ax.set_xticklabels(residue_labels)
    ax.set_ylabel("Average Distance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

####################################################################################################################
# 239 to 320
skip_pairs = [
    ("HB2", "HG11"), ("HB2", "HG12"), ("HB2", "HG13"),
    ("HB3", "HG11"), ("HB3", "HG12"), ("HB3", "HG13"),
    ("HB2", "HG21"), ("HB2", "HG22"), ("HB2", "HG23"),
    ("HB3", "HG21"), ("HB3", "HG22"), ("HB3", "HG23")
]
wt_320 = average_per_file_with_keywords(file_path, ["wt","239","320"], skip_pairs = skip_pairs)
mut_320 = average_per_file_with_keywords(file_path, ["mut","239","320"], skip_pairs = skip_pairs)
plot_wt_mut_comparison(wt_320, mut_320, title = "WT vs MUT for 239-320")

# 239 to 321
wt_321 = average_per_file_with_keywords(file_path, ["wt", "321"])
mut_321 = average_per_file_with_keywords(file_path, ["mut", "321"])
plot_wt_mut_comparison(wt_321, mut_321, title = "WT vs MUT for 239-321")

# 239 to 322
wt_322 = average_per_file_with_keywords(file_path, ["wt", "322"])
mut_322 = average_per_file_with_keywords(file_path, ["mut", "322"])
plot_wt_mut_comparison(wt_322, mut_322, title = "WT vs MUT for 239-322")

# 239 to 679
skip_pairs = [
    ("HD21", "HZ1"), ("HD21", "HZ2"), ("HD21", "HZ3")
]
wt_679 = average_per_file_with_keywords(file_path, ["wt", "679"])
mut_679 = average_per_file_with_keywords(file_path, ["mut", "679"], skip_pairs=skip_pairs)
plot_wt_mut_comparison(wt_679, mut_679, title = "WT vs MUT for 239-679")

# ADP
wt_ADP = average_per_file_with_keywords(file_path, ["wt", "852"])
mut_ADP = average_per_file_with_keywords(file_path, ["mut", "852"])
plot_wt_mut_comparison(wt_ADP, mut_ADP, title = "WT vs MUT for ADP")


# Combined Residues to Residues
plot_combined_wt_mut_comparisons(
    wt_dicts=[wt_320, wt_321, wt_322, wt_679],
    mut_dicts=[mut_320, mut_321, mut_322, mut_679],
    residue_labels=["239-320","239-321", "239-322", "239-679"],
    title="WT vs MUT Distance Comparisons Across Residue Interactions"
)