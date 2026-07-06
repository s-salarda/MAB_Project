#!/usr/bin/env python3.12
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================
GENOTYPES = ['WT', 'D239N', 'K637E']
SIMS = [1, 2, 3]
BINDING_DIR = "D:\\Projects\\MAB_project\\CM_Loop2_PCA_Analysis\\kalen_csv\\csv\\"
BASE_DIR = "D:\\Projects\\MAB_project\\CM_Loop2"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Reference point (Actin N-terminus coordinates)
ACTIN_NTERM = {
    'x': -57.33325,
    'y': 4.03525,
    'z': 563.8495
}

# Colors for plotting
COLORS = {
    'WT': 'black',
    'D239N': 'blue',
    'K637E': 'red'
}

# =============================================================================
# DATA READING FUNCTIONS
# =============================================================================
def read_vector_file(filepath, genotype, sim_number):
    """
    Read a cpptraj vector output file.
    """
    # Read the file (skip first line which is the header)
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=1, 
                     names=['Frame', 'VectorX', 'VectorY', 'VectorZ'])
    
    # Rename and process columns
    df['Genotype'] = genotype
    df['sim'] = sim_number
    df['frame'] = df['Frame'].astype(float)
    df['x'] = df['VectorX'].astype(float)
    df['y'] = df['VectorY'].astype(float)
    df['z'] = df['VectorZ'].astype(float)
    
    # Keep only the processed columns
    df = df[['Genotype', 'sim', 'frame', 'x', 'y', 'z']]
    return df


def load_all_vector_data(base_dir, genotypes, sims):
    """
    Load all vector data files for all genotypes and simulations.
    """
    genotype_dfs = {}
    
    for genotype in genotypes:
        sim_list = []
        
        for sim in sims:
            # Construct file path
            filepath = os.path.join(base_dir, f"{genotype}_sim{sim}_vector.dat")
            
            # Read and store
            df = read_vector_file(filepath, genotype, sim)
            sim_list.append(df)
            print(f"Loaded: {genotype} sim{sim} - {len(df)} frames")
        
        # Combine sims for this genotype
        genotype_dfs[genotype] = pd.concat(sim_list, ignore_index=True)
        print(f"{genotype}: {len(genotype_dfs[genotype])} total frames")
    
    return genotype_dfs


# =============================================================================
# DISTANCE CALCULATION FUNCTIONS
# =============================================================================
def calculate_distance(df, reference_point):
    """
    Calculate Euclidean distance from x,y,z coordinates to a reference point.

    """
    df['distance'] = np.sqrt(
        (df['x'] - reference_point['x'])**2 + 
        (df['y'] - reference_point['y'])**2 + 
        (df['z'] - reference_point['z'])**2
    )
    return df


def combine_all_genotypes(genotype_dfs, genotypes):
    """
    Combine all genotype dataframes into one master dataframe.
    """
    df_all = pd.concat(genotype_dfs.values(), ignore_index=True)
    
    # Make Genotype categorical with specific order
    df_all['Genotype'] = pd.Categorical(
        df_all['Genotype'], 
        categories=genotypes, 
        ordered=True
    )
    
    return df_all


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================
def perform_anova(genotype_dfs, genotypes):
    """
    Perform one-way ANOVA on distance data across genotypes.
    """
    data_arrays = [genotype_dfs[g]['distance'].values for g in genotypes]
    f_stat, p_value = stats.f_oneway(*data_arrays)
    
    print(f"\n--- ANOVA Results ---")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    return f_stat, p_value


def perform_pairwise_tests(genotype_dfs, genotypes):
    """
    Perform pairwise t-tests with Bonferroni correction.
    """
    comparisons = []
    for i in range(len(genotypes)):
        for j in range(i+1, len(genotypes)):
            comparisons.append((genotypes[i], genotypes[j]))
    
    bonferroni_alpha = 0.05 / len(comparisons)
    print(f"\n--- Pairwise t-tests (Bonferroni α = {bonferroni_alpha:.4f}) ---")
    
    pairwise_results = []
    
    for g1, g2 in comparisons:
        data1 = genotype_dfs[g1]['distance'].values
        data2 = genotype_dfs[g2]['distance'].values
        
        t_stat, p_val = stats.ttest_ind(data1, data2)
        
        # Calculate Cohen's d (effect size)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        n1, n2 = len(data1), len(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        sig = "***" if p_val < bonferroni_alpha else "ns"
        print(f"{g1} vs {g2}: t={t_stat:.4f}, p={p_val:.4e}, d={cohens_d:.4f} {sig}")
        
        pairwise_results.append({
            'Comparison': f"{g1} vs {g2}",
            'Mean_1': mean1,
            'Mean_2': mean2,
            'Mean_Difference': mean1 - mean2,
            't_statistic': t_stat,
            'p_value': p_val,
            'Cohens_d': cohens_d,
            'Significant': sig
        })
    
    return pd.DataFrame(pairwise_results)


def calculate_summary_statistics(df_all):
    """
    Calculate summary statistics for each genotype.
    """
    summary = df_all.groupby('Genotype')['distance'].agg([
        ('Mean', 'mean'),
        ('SD', 'std'),
        ('SEM', lambda x: x.std() / np.sqrt(len(x))),
        ('N', 'count')
    ])
    
    print("\n--- Summary Statistics ---")
    print(summary)
    
    return summary


def calculate_detailed_statistics(df_all):
    """
    Calculate detailed statistics by genotype and simulation.
    """
    detailed_stats = df_all.groupby(['Genotype', 'sim'])['distance'].agg([
        ('Mean', 'mean'),
        ('SD', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Median', 'median'),
        ('N', 'count')
    ])
    
    return detailed_stats


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_violin_boxplot(genotype_dfs, genotypes, colors, results_dir):
    """
    Create violin plot with box plot overlay.
    """
    print("\n--- Creating Graph 1: Violin Plot ---")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_arrays = [genotype_dfs[g]['distance'].values for g in genotypes]
    
    # Create violin plot
    parts = ax.violinplot(
        data_arrays,
        positions=range(1, len(genotypes)+1),
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Customize violin colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[genotypes[i]])
        pc.set_alpha(0.7)
    
    # Add box plot overlay
    bp = ax.boxplot(
        data_arrays,
        positions=range(1, len(genotypes)+1),
        widths=0.3,
        patch_artist=True,
        showfliers=False
    )
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[genotypes[i]])
        patch.set_alpha(0.3)
    
    ax.set_xticks(range(1, len(genotypes)+1))
    ax.set_xticklabels(genotypes)
    ax.set_ylabel('Distance (Å)', fontsize=12)
    ax.set_xlabel('Genotype', fontsize=12)
    ax.set_title('CM Loop to Actin N-terminus Distance', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'distance_violin_plot.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_distance_vs_frame(df_all, genotypes, colors, results_dir):
    """
    Create time series plot of distance vs frame.
    """
    print("\n--- Creating Graph 2: Distance over Frames ---")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for genotype in genotypes:
        df_genotype = df_all[df_all['Genotype'] == genotype]
        
        # Plot each simulation as a separate line with transparency
        for sim in df_genotype['sim'].unique():
            df_sim = df_genotype[df_genotype['sim'] == sim]
            ax.plot(df_sim['frame'], df_sim['distance'], 
                   color=colors[genotype], alpha=0.5, linewidth=0.8)
        
        # Calculate and plot the mean across all simulations
        mean_by_frame = df_genotype.groupby('frame')['distance'].mean()
        ax.plot(mean_by_frame.index, mean_by_frame.values, 
               color=colors[genotype], linewidth=2, label=genotype)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Distance (Å)', fontsize=12)
    ax.set_title('Distance vs Frame (all simulations)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'distance_vs_frame.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_average_barplot(df_all, genotypes, colors, results_dir):
    """
    Create bar plot of average distances with error bars.
    """
    print("\n--- Creating Graph 3: Bar Graph of Averages ---")
    
    # Calculate means and SEMs
    means = []
    sems = []
    
    for genotype in genotypes:
        data = df_all[df_all['Genotype'] == genotype]['distance']
        means.append(data.mean())
        sems.append(data.std() / np.sqrt(len(data)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bar plot
    x_pos = np.arange(len(genotypes))
    bars = ax.bar(x_pos, means, yerr=sems, 
                  color=[colors[g] for g in genotypes],
                  alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genotypes, fontsize=12)
    ax.set_ylabel('Average Distance (Å)', fontsize=12)
    ax.set_xlabel('Genotype', fontsize=12)
    ax.set_title('Average CM Loop to Actin N-terminus Distance', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, sem) in enumerate(zip(means, sems)):
        ax.text(i, mean + sem + 0.5, f'{mean:.2f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'average_distance_barplot.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


# =============================================================================
# BINDING FORCE ANALYSIS FUNCTIONS
# =============================================================================
def load_binding_force_data(force_file):
    """
    Load binding force data from CSV file.
    """
    try:
        if os.path.exists(force_file):
            df_force = pd.read_csv(force_file)
            print(f"\n--- Loaded Binding Force Data ---")
            print(f"Loaded {len(df_force)} binding force records")
            return df_force
        else:
            print(f"\nBinding force file not found at: {force_file}")
            return None
    except Exception as e:
        print(f"Error loading binding force data: {e}")
        return None


def plot_binding_force_barplot(df_force, genotypes, colors, results_dir):
    """
    Create bar plot of average binding forces.
    """
    print("\n--- Creating Binding Force Bar Plot ---")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    force_means = []
    force_sems = []
    
    for genotype in genotypes:
        data = df_force[df_force['Genotype'] == genotype]['binding_force']
        force_means.append(data.mean())
        force_sems.append(data.std() / np.sqrt(len(data)))
        print (force_means, force_sems)
    
    x_pos = np.arange(len(genotypes))
    bars = ax.bar(x_pos, force_means, yerr=force_sems,
                  color=[colors[g] for g in genotypes],
                  alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genotypes, fontsize=12)
    ax.set_ylabel('Average Binding Force', fontsize=12)
    ax.set_xlabel('Genotype', fontsize=12)
    ax.set_title('Average Binding Force by Genotype', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, sem) in enumerate(zip(force_means, force_sems)):
        ax.text(i, mean + sem + 0.2, f'{mean:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'binding_force_barplot.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_binding_force_vs_time(df_force, genotypes, colors, results_dir):
    """
    Create time series plot of binding force vs time.
    """
    print("\n--- Creating Binding Force vs Time Plot ---")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for genotype in genotypes:
        df_genotype = df_force[df_force['Genotype'] == genotype]
        
        for sim in df_genotype['sim'].unique():
            df_sim = df_genotype[df_genotype['sim'] == sim]
            if len(df_sim) > 0:
                ax.plot(df_sim['time'], df_sim['binding_force'],
                       color=colors[genotype], alpha=0.5, linewidth=0.8)
        
        # Mean line
        mean_by_time = df_genotype.groupby('time')['binding_force'].mean()
        ax.plot(mean_by_time.index, mean_by_time.values,
               color=colors[genotype], linewidth=2, label=genotype)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Binding Force', fontsize=12)
    ax.set_title('Binding Force vs Time (all simulations)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'binding_force_vs_time.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def calculate_binding_force_statistics(df_force):
    """
    Calculate summary statistics for binding force data.
    """
    # Overall summary
    force_summary = df_force.groupby('Genotype')['binding_force'].agg([
        ('Mean', 'mean'),
        ('SD', 'std'),
        ('SEM', lambda x: x.std() / np.sqrt(len(x))),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Median', 'median'),
        ('N', 'count')
    ]).round(2)
    
    # Detailed by simulation
    force_detailed = df_force.groupby(['Genotype', 'sim'])['binding_force'].agg([
        ('Mean', 'mean'),
        ('SD', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Median', 'median'),
        ('N', 'count')
    ]).round(2)
    
    return force_summary, force_detailed


# =============================================================================
# DATA EXPORT FUNCTIONS
# =============================================================================
def export_distance_data(df_all, summary, detailed_stats, results_dir):
    """
    Export distance analysis data to CSV files.
    """
    print("\n--- Exporting Distance Data to CSV ---")
    
    # Export complete combined dataframe
    filepath = os.path.join(results_dir, 'all_distances_combined.csv')
    df_all.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")
    
    # Export summary statistics
    filepath = os.path.join(results_dir, 'distance_summary_statistics.csv')
    summary.to_csv(filepath)
    print(f"Saved: {filepath}")
    
    # Export detailed statistics
    filepath = os.path.join(results_dir, 'distance_stats_by_sim.csv')
    detailed_stats.to_csv(filepath)
    print(f"Saved: {filepath}")


def export_statistical_tests(f_stat, p_value, df_pairwise, results_dir):
    """
    Export ANOVA and pairwise test results to CSV files.
    """
    print("\n--- Exporting Statistical Test Results ---")
    
    # Export ANOVA results
    anova_results = pd.DataFrame({
        'Test': ['One-way ANOVA'],
        'F_statistic': [f_stat],
        'p_value': [p_value],
        'Significant': ['Yes' if p_value < 0.05 else 'No']
    })
    filepath = os.path.join(results_dir, 'anova_results.csv')
    anova_results.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")
    
    # Export pairwise test results
    filepath = os.path.join(results_dir, 'pairwise_tests.csv')
    df_pairwise.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")


def export_binding_force_data(force_summary, force_detailed, results_dir):
    """
    Export binding force statistics to CSV files.
    """
    print("\n--- Exporting Binding Force Data ---")
    
    filepath = os.path.join(results_dir, 'binding_force_summary.csv')
    force_summary.to_csv(filepath)
    print(f"Saved: {filepath}")
    
    filepath = os.path.join(results_dir, 'binding_force_by_sim.csv')
    force_detailed.to_csv(filepath)
    print(f"Saved: {filepath}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """
    Main function to run the complete analysis pipeline.
    """
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\nResults will be saved to: {RESULTS_DIR}")
    
    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    genotype_dfs = load_all_vector_data(BASE_DIR, GENOTYPES, SIMS)
    
    # -------------------------------------------------------------------------
    # 2. CALCULATE DISTANCES
    # -------------------------------------------------------------------------
    for genotype in GENOTYPES:
        genotype_dfs[genotype] = calculate_distance(
            genotype_dfs[genotype], 
            ACTIN_NTERM
        )
    
    df_all = combine_all_genotypes(genotype_dfs, GENOTYPES)
    print(f"\nCombined dataframe: {len(df_all)} total observations")
    
    # -------------------------------------------------------------------------
    # 3. STATISTICAL ANALYSIS
    # -------------------------------------------------------------------------
    # ANOVA
    f_stat, p_value = perform_anova(genotype_dfs, GENOTYPES)
    
    # Pairwise tests
    df_pairwise = perform_pairwise_tests(genotype_dfs, GENOTYPES)
    
    # Summary statistics
    summary = calculate_summary_statistics(df_all)
    detailed_stats = calculate_detailed_statistics(df_all)
    
    # -------------------------------------------------------------------------
    # 4. EXPORT DISTANCE DATA
    # -------------------------------------------------------------------------
    export_distance_data(df_all, summary, detailed_stats, RESULTS_DIR)
    export_statistical_tests(f_stat, p_value, df_pairwise, RESULTS_DIR)
    
    # -------------------------------------------------------------------------
    # 5. CREATE DISTANCE PLOTS
    # -------------------------------------------------------------------------
    plot_violin_boxplot(genotype_dfs, GENOTYPES, COLORS, RESULTS_DIR)
    plot_distance_vs_frame(df_all, GENOTYPES, COLORS, RESULTS_DIR)
    plot_average_barplot(df_all, GENOTYPES, COLORS, RESULTS_DIR)
    
    # -------------------------------------------------------------------------
    # 6. BINDING FORCE ANALYSIS (if data available)
    # ------------------------------------------------------------------------- 
    force_file = os.path.join(BINDING_DIR, "results_formatted.csv")
    df_force = load_binding_force_data(force_file)
    
    if df_force is not None:
        # Create plots
        plot_binding_force_barplot(df_force, GENOTYPES, COLORS, RESULTS_DIR)
        plot_binding_force_vs_time(df_force, GENOTYPES, COLORS, RESULTS_DIR)
        
        # Calculate and export statistics
        force_summary, force_detailed = calculate_binding_force_statistics(df_force)
        export_binding_force_data(force_summary, force_detailed, RESULTS_DIR)


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()