# RMSF Analysis ----------------------------------------------------------------
library(tidyverse)
library(dplyr)
library(purrr)
library(ggplot2)
library(ggpubr)

# Colors -----------------------------------------------------------------------
genotype_color_bar <- c(
  "WT" = "darkgrey",    # WT
  "D239N" = "green",    # d239n
  "K637E" = "blue"      # k637e
)

genotype_color_ticks <- c(
  "WT" = "black",    # WT
  "D239N" = "darkgreen",    # d239n
  "K637E" = "darkblue"      # k637e
)

sim_colors <- c(
  "WT_1"   = "black",     "WT_2"   = "darkgray",  "WT_3"   = "lightgray",
  "D239N_1"= "darkgreen", "D239N_2"= "green",     "D239N_3"= "lightgreen",
  "K637E_1"= "darkblue",  "K637E_2"= "blue",      "K637E_3"= "lightblue"
)

# Genotype Files
wt_files <- c("results/wt1_rmsf.dat", "results/wt2_rmsf.dat",
              "results/wt3_rmsf.dat")
d239n_files <- c("results/d239n1_rmsf.dat", "results/d239n2_rmsf.dat", 
                 "results/d239n3_rmsf.dat")
k637e_files <- c("results/k637e1_rmsf.dat", "results/k637e2_rmsf.dat",
                 "results/k637e3_rmsf.dat")
all_files <- c(wt_files, d239n_files, k637e_files)

# Create a combined df of all Genotypes ---------------------------------------
plots_file <- 
  
# Add column for genotype and sim
df_all <- all_files %>%
  set_names() %>%   # keeps filenames as IDs
  map_dfr(~read_table(.x, col_names = FALSE, skip = 1), .id = "file") %>%
  rename(residue = X1, rmsf = X2) %>%
  mutate(
    file = basename(file),
    residue = as.integer(residue)
  ) %>%
  mutate(
    # Extract genotype name (wt1_rmsf.dat -> wt, d239n2_rmsf.dat -> d239n)
    genotype = toupper(sub("([0-9]_rmsf\\.dat)", "", file)),
    # Extract simulation number (the digit right before _rmsf.dat)
    sim = as.integer(str_extract(file, "(?<=[:alpha:])[0-9](?=_rmsf)")),
    # Convert genotype to factor
    genotype = factor(genotype, levels = c("WT", "D239N", "K637E"))
  ) %>%
  mutate(
    # Create combined identifier for coloring
    sim_id = paste0(genotype, "_", sim),
    # Convert to factor with levels matching sim_colors
    sim_id = factor(sim_id, levels = c("WT_1", "WT_2", "WT_3",
                                       "D239N_1", "D239N_2", "D239N_3",
                                       "K637E_1", "K637E_2", "K637E_3"))
  )

# Create plots directory if it doesn't exist
dir.create("plots", showWarnings = FALSE)
dir.create("plots/line_plots", showWarnings = FALSE)
dir.create("plots/bar_plots", showWarnings = FALSE)

# Graph RMSF -------------------------------------------------------------------
# All sims on a plot
ggplot() +
  geom_line(data = df_all,
            aes(x = residue, y = rmsf,
                color = sim_id, group = sim_id),
            linewidth = 0.5) +
  
  scale_color_manual(values = sim_colors, 
                     name = "Simulation",
                     breaks = names(sim_colors)) +
  
  labs(title = "RMSF of All Simulations and Residues of Interest",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text = element_text(size = 10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top", legend.box.just = "center") +
  guides(color = guide_legend(nrow = 1, ncol = 9, byrow = TRUE))

ggsave("plots/line_plots/all_sims_rmsf.png",
       width = 20, height = 10, dpi = 500)

# All sims faceted by genotype in 3x3 grid
ggplot() +
  geom_line(data = df_all,
            aes(x = residue, y = rmsf,
                color = sim_id, group = sim_id),
            linewidth = 0.5) +
  
  scale_color_manual(values = sim_colors, 
                     name = "Simulation",
                     breaks = names(sim_colors)) +
  
  facet_wrap(~sim_id, ncol = 3) +
  
  labs(title = "RMSF - All Simulations in 3x3 Grid",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text = element_text(size = 10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top", legend.box.just = "center") +
  guides(color = guide_legend(nrow = 1, ncol = 9, byrow = TRUE))

ggsave("plots/line_plots/all_sims_rmsf_3x3.png",
       width = 20, height = 15, dpi = 500)

# RMSF by genotype - each genotype in own panel
ggplot() +
  geom_line(data = df_all,
            aes(x = residue, y = rmsf,
                color = sim_id, group = sim_id)) +
  
  scale_color_manual(values = sim_colors, name = "Simulation") +
  
  facet_wrap(~genotype, ncol = 3, scales = "free_y") +
  
  labs(title = "RMSF by Genotype",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text = element_text(size = 10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        legend.position = "top", legend.box.just = "center") +
  guides(color = guide_legend(nrow = 1, ncol = 9, byrow = TRUE))

ggsave("plots/line_plots/rmsf_by_genotype.png",
       width = 20, height = 6, dpi = 500)

# Calculate average RMSF
df_avg <- df_all %>%
  group_by(genotype, residue) %>%
  summarise(
    mean_rmsf = mean(rmsf),
    sd_rmsf = sd(rmsf),
    se_rmsf = sd(rmsf) / sqrt(n()),
    .groups = "drop"
  )

# Average RMSF by genotype with SEM
ggplot() +
  geom_line(data = df_avg,
            aes(x = residue, y = mean_rmsf,
                color = genotype), linewidth = 0.8) +
  geom_ribbon(data = df_avg,
              aes(x = residue, y = mean_rmsf,
                  ymin = mean_rmsf - se_rmsf, ymax = mean_rmsf + se_rmsf,
                  fill = genotype),
              alpha = 0.3, color = NA) +
  
  scale_color_manual(values = genotype_color_bar) +
  scale_fill_manual(values = genotype_color_bar) +
  
  facet_wrap(~genotype, ncol = 3, scales = "free_y") +
  
  labs(title = "Average RMSF by Genotype (Mean ± SE)",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text = element_text(size = 10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        legend.position = "top", legend.box.just = "center") +
  guides(color = guide_legend(nrow = 1, ncol = 9, byrow = TRUE))

ggsave("plots/line_plots/average_rmsf_by_genotype.png",
       width = 20, height = 8, dpi = 500)

# Bar graph for specific residue ranges
# Define regions of interest
regions <- list(
  "Loop 2" = c(624, 647),
  "S1" = c(232,244),
  "HLH" = c(520,556),
  "O-Helix" = c(479,503),
  "Relay Helix" = c(479,503),
  "SH1" = c(232,244),
  "S2" = c(462,472),
  "Loop 3" = c(567,577),
  "Loop 4" = c(361,378),
  "P Loop" = c(179,183),
  "Purine Loop" = c(126, 131),
  "ADP.Pi" = c(852,853)
)

# Filter data for regions and create bar plot for each region
for (region_name in names(regions)) {
  region_range <- regions[[region_name]]
  start_res <- region_range[1]
  end_res <- region_range[2]
  
  # Filter data for this region
  df_region <- df_all %>%
    filter(residue >= start_res & residue <= end_res)
  
  # Create bar plot
  ggplot() +
    geom_col(data = df_region,
             aes(x = as.factor(residue), y = rmsf,
                 fill = sim_id), position = "dodge") +
    
    scale_fill_manual(values = sim_colors, name = "Simulation") +
    
    labs(title = paste0("RMSF for ", region_name, " (Residues ", start_res, "-", end_res, ")"),
         x = "Residue", y = "RMSF (Å)") +
    
    theme_pubr() +
    theme(text = element_text(size = 10, face = "bold"), 
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top", legend.box.just = "center") +
    guides(fill = guide_legend(nrow = 1, ncol = 9, byrow = TRUE))
  
  # Save plot
  filename <- paste0("plots/bar_plots/rmsf_bar_", gsub(" ", "_", tolower(region_name)), ".png")
  ggsave(filename, width = 16, height = 8, dpi = 500)
  
  # Filter the already-calculated df_avg for this region
  df_region_avg <- df_avg %>%
    filter(residue >= start_res & residue <= end_res)
  
  # Create average bar plot with SEM and individual points
  ggplot() +
    geom_col(data = df_region_avg,
             aes(x = as.factor(residue), y = mean_rmsf,
                 fill = genotype),
             position = position_dodge(width = 0.7), width = 0.6) +
    
    geom_errorbar(data = df_region_avg,
                  aes(x = as.factor(residue),
                      ymin = mean_rmsf - se_rmsf, ymax = mean_rmsf + se_rmsf,
                      group = genotype),
                  position = position_dodge(width = 0.7), width = 0.3) +
    
    geom_point(data = df_region,
               aes(x = as.factor(residue), y = rmsf,
                   group = genotype, fill = genotype,
                   color = genotype),
               position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.7), # Tick jitters
               size = 1.5, shape = 21, stroke = 1,
               show.legend = FALSE)+
    # This might not be right
    stat_compare_means(
      data = df_region,
      aes(x = as.factor(residue), y = rmsf, group = genotype),
      method = "anova",
      label = "p.signif", # *** p<0.001, ** P<0.01, * p<0.05
      size = 10,
      label.y = max(df_region_avg$mean_rmsf + df_region_avg$se_rmsf) * 1.07,
      inherit.aes = FALSE,
      hide.ns = TRUE) +
    
    scale_fill_manual(values = genotype_color_bar, name = "Genotype") +
    scale_color_manual(values = genotype_color_ticks) +
    
    labs(title = paste0("Average RMSF for ", region_name, " (Residues ", start_res, "-", end_res, ")"),
         x = "Residue", y = "RMSF (Å)") +
    
    theme_pubr() +
    theme(text = element_text(size = 10, face = "bold"), 
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top", legend.box.just = "center")+
  guides(fill = guide_legend(nrow = 1, ncol = 9, byrow = TRUE))
  
  # Save average plot
  filename_avg <- paste0("plots/bar_plots/rmsf_bar_avg_", gsub(" ", "_", tolower(region_name)), ".png")
  ggsave(filename_avg, width = 20, height = 8, dpi = 500)
}

print("Analysis complete! Plots saved in 'plots/' directory.")
