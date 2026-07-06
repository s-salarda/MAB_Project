# RMSF Analysis — WT vs D239N --------------------------------------------------
library(tidyverse)
library(dplyr)
library(purrr)
library(ggplot2)
library(ggpubr)

# Colors -----------------------------------------------------------------------
genotype_color_bar <- c(
  "WT"    = "darkgrey",
  "D239N" = "green"
)

genotype_color_ticks <- c(
  "WT"    = "black",
  "D239N" = "darkgreen"
)

sim_colors <- c(
  "WT_1"    = "black",     "WT_2"    = "darkgray",   "WT_3"    = "lightgray",
  "D239N_1" = "darkgreen", "D239N_2" = "green",      "D239N_3" = "lightgreen"
)

# Domain background shading colors --------------------------------------------
# Colors are matching to Chimera color scripts
domain_colors <- c(
  "P Loop"      = "#E8E8FF",
  "Purine Loop" = "#E8E8FF",
  "SH1"         = "#9090CF",
  "S1"          = "#FF9999",
  "S2"          = "#FFB899",
  "Loop 2"      = "#E07050",
  "Loop 3"      = "#E07050",
  "Upper 50kDa" = "#A8E4EC",
  "Lower 50kDa" = "#90C4E8",
  "ADP.Pi"      = "#90E0A0",
  "Loop 4"      = "#E07050"
)

# Genotype Files ---------------------------------------------------------------
wt_files <- c("results/align_cut_ploop/wt1_rmsf.dat", 
              "results/align_cut_ploop/wt2_rmsf.dat",
              "results/align_cut_ploop/wt3_rmsf.dat")
d239n_files <- c("results/align_cut_ploop/d239n1_rmsf.dat", 
                 "results/align_cut_ploop/d239n2_rmsf.dat",
                 "results/align_cut_ploop/d239n3_rmsf.dat")
all_files <- c(wt_files, d239n_files)

# Create a combined df ---------------------------------------------------------
# .dat files column headers: #Res  AtomicFlux
df_all <- all_files %>%
  set_names() %>%
  # read each file, skip header row (skip = 1_, puts everything to one df
  # setnames() = full file path is saved in file column
  map_dfr(~read_table(.x, col_names = FALSE, skip = 1), .id = "file") %>%
  # X1 = #res X2 = AtomixFlux
  rename(residue = X1, rmsf = X2) %>%
  mutate(
    file    = basename(file),
    residue = as.integer(residue)
  ) %>%
  # Adds/fixes genotype and sim columns
  mutate(
    genotype = toupper(sub("([0-9]_rmsf\\.dat)", "", file)),
    sim = as.integer(str_extract(file, "(?<=[a-zA-Z])[0-9](?=_rmsf)")),
    genotype = factor(genotype, levels = c("WT", "D239N"))
  ) %>%
  # adds straight forward sim ID to filter
  mutate(
    sim_id = paste0(genotype, "_", sim),
    sim_id = factor(sim_id, levels = c("WT_1", "WT_2", "WT_3",
                                       "D239N_1", "D239N_2", "D239N_3"))
  )

# Create plots directory -------------------------------------------------------
dir.create("plots", showWarnings = FALSE)
dir.create("plots/line_plots", showWarnings = FALSE)
dir.create("plots/bar_plots", showWarnings = FALSE)

# Graphs -------------------------------------------------------
# Graph all sims along a line plot
ggplot() +
  geom_line(data = df_all,
            aes(x = residue, y = rmsf,
                color = sim_id, group = sim_id),
            linewidth = 0.8) +
  
  scale_color_manual(values = sim_colors,
                     name   = "Simulation",
                     breaks = names(sim_colors)) +
  
  labs(title = "RMSF of All Simulations",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text           = element_text(size = 18, face = "bold"),
        plot.title     = element_text(hjust = 0.5),
        axis.text.x    = element_text(angle = 0, hjust = 0.5),
        legend.position = "top", legend.box.just = "center") +
  guides(color = guide_legend(nrow = 1, ncol = 6, byrow = TRUE))

ggsave("plots/line_plots/all_sims_rmsf_wt_d239n.png",
       width = 20, height = 10, dpi = 500)

# Separate all line plot sims to their own panel
ggplot() +
  geom_line(data = df_all,
            aes(x = residue, y = rmsf,
                color = sim_id, group = sim_id),
            linewidth = 0.8) +
  
  scale_color_manual(values = sim_colors,
                     name   = "Simulation",
                     breaks = names(sim_colors)) +
  
  # Splits to a 2x3 grid
  facet_wrap(~sim_id, ncol = 3) +
  
  labs(title = "RMSF of all Simulations",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text           = element_text(size = 18, face = "bold"),
        plot.title     = element_text(hjust = 0.5),
        axis.text.x    = element_text(angle = 0, hjust = 0.5),
        legend.position = "top", legend.box.just = "center") +
  guides(color = guide_legend(nrow = 1, ncol = 6, byrow = TRUE))

ggsave("plots/line_plots/separate_all_sims_rmsf_wt_d239n.png",
       width = 20, height = 10, dpi = 500)

# Separate sims but grouped by Genotype
ggplot() +
  geom_line(data = df_all,
            aes(x = residue, y = rmsf,
                color = sim_id, group = sim_id),
            linewidth = 0.8) +
  
  scale_color_manual(values = sim_colors, name = "Simulation") +
  
  # "free_y scales y independently for each panel
  facet_wrap(~genotype, ncol = 2, scales = "free_y") +
  
  labs(title = "RMSF of all Simulation by Genotype",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text           = element_text(size = 18, face = "bold"),
        plot.title     = element_text(hjust = 0.5),
        axis.text.x    = element_text(angle = 0, hjust = 0.5),
        legend.position = "top", legend.box.just = "center") +
  guides(color = guide_legend(nrow = 1, ncol = 6, byrow = TRUE))

ggsave("plots/line_plots/rmsf_by_genotype_wt_d239n.png",
       width = 20, height = 6, dpi = 500)

# Calculate average RMSF -------------------------------------------------------
df_avg <- df_all %>%
  group_by(genotype, residue) %>%
  summarise(
    mean_rmsf = mean(rmsf),
    sd_rmsf   = sd(rmsf),
    se_rmsf   = sd(rmsf) / sqrt(n()),
    
    # removes grouping structure so its just a df
    .groups   = "drop"
  )

# Average RMSF by genotype with SEM 
ggplot() +
  geom_line(data = df_avg,
            aes(x = residue, y = mean_rmsf,
                color = genotype), 
            linewidth = 0.8) +
  
  geom_ribbon(data = df_avg,
              aes(x = residue, y = mean_rmsf,
                  ymin = mean_rmsf - se_rmsf,
                  ymax = mean_rmsf + se_rmsf,
                  fill = genotype),
              alpha = 0.3, color = NA) +
  
  scale_color_manual(values = genotype_color_bar) +
  scale_fill_manual(values  = genotype_color_bar) +
  
  facet_wrap(~genotype, ncol = 2, scales = "free_y") +
  
  labs(title = "Average RMSF by Genotype (Mean ± SE)",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text           = element_text(size = 18, face = "bold"),
        plot.title     = element_text(hjust = 0.5),
        axis.text.x    = element_text(angle = 0, hjust = 0.5),
        legend.position = "top", legend.box.just = "center") +
  guides(color = guide_legend(nrow = 1, ncol = 6, byrow = TRUE))

ggsave("plots/line_plots/average_rmsf_by_genotype_wt_d239n.png",
       width = 20, height = 8, dpi = 500)

# Bar plots per region — WT vs D239N with t-test ------------------------------
# Domain ranges for regions in myosin
regions <- list(
  "Loop 2"     = c(624, 647),
  "S1"         = c(232, 244),
  "HLH"        = c(520, 556),
  "O-Helix"    = c(479, 503),
  "Relay Helix"= c(479, 503),
  "SH1"        = c(674, 685),
  "S2"         = c(462, 472),
  "Loop 3"     = c(567, 577),
  "Loop 4"     = c(361, 378),
  "P Loop"     = c(179, 183),
  "Purine Loop"= c(129, 131),
  "ADP.Pi"     = c(852, 853)
)

# For loop to create bar plots for each region ---------------------------------
# OVERWRITES df_region and df_region_avg!!
for (region_name in names(regions)) {
  region_range <- regions[[region_name]]
  start_res    <- region_range[1]
  end_res      <- region_range[2]
  
  df_region     <- df_all %>% filter(residue >= start_res & residue <= end_res)
  df_region_avg <- df_avg %>% filter(residue >= start_res & residue <= end_res)
  
  if (nrow(df_region) == 0) next
  
  # Check to make sure its going over each region
  # n rows in df_region = n residues * 6 sims
  cat("Processing region:", region_name, 
      "| Residues:", start_res, "-", end_res,
      "| n residues:", length(start_res:end_res),
      "| Residues:", start_res, "-", end_res,
      "| n residues x 6:", length(start_res:end_res) * 6,
      "| n rows in df_region:", nrow(df_region), "\n")
  
  # Bar Plot function for the RMSF of all sims
  ggplot() +
    geom_col(data = df_region,
             aes(x = as.factor(residue), y = rmsf,
                 fill = sim_id),
             position = position_dodge(width = 0.8), width = 0.80,
             colour = "black", linewidth = 0.4) +
    
    scale_fill_manual(values = sim_colors, name = "Simulation") +
    
    labs(title = paste0("RMSF for ", region_name,
                        " (Residues ", start_res, "-", end_res, ")"),
         x = "Residue", y = "RMSF (Å)") +
    
    theme_pubr() +
    theme(text           = element_text(size = 18, face = "bold"),
          plot.title     = element_text(hjust = 0.5),
          axis.text.x    = element_text(angle = 45, hjust = 1),
          legend.position = "top", legend.box.just = "center") +
    guides(fill = guide_legend(nrow = 1, ncol = 6, byrow = TRUE))
  
  filename <- paste0("plots/bar_plots/rmsf_bar_wt_d239n_",
                     gsub(" ", "_", tolower(region_name)), ".png")
  ggsave(filename, width = 16, height = 8, dpi = 500)
  
  # Genotype Averaged bar plot with SEM and sim ticks
  ggplot() +
    geom_col(data     = df_region_avg,
             aes(x = as.factor(residue), y = mean_rmsf, fill = genotype),
             position = position_dodge(width = 0.9), width = 0.80,
             colour = "black", linewidth = 0.4) +
    
    geom_errorbar(data = df_region_avg,
                  aes(x     = as.factor(residue),
                      ymin  = mean_rmsf - se_rmsf,
                      ymax  = mean_rmsf + se_rmsf,
                      group = genotype),
                  position  = position_dodge(width = 0.9),
                  width     = 0.2) +
    
    geom_point(data = df_region,
               aes(x = as.factor(residue), y = rmsf,
                   group = genotype, fill = genotype, color = genotype),
               position    = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.9),
               size        = 1.5, shape = 21, stroke = 1,
               show.legend = FALSE) +
    
    # Runs a t-test and adds the p val stars 
    stat_compare_means(
      data        = df_region,
      aes(x = as.factor(residue), y = rmsf, group = genotype),
      method      = "t.test",
      label       = "p.signif",
      size        = 10,
      label.y     = max(df_region_avg$mean_rmsf + df_region_avg$se_rmsf) * 1.07,
      inherit.aes = FALSE,
      hide.ns     = TRUE) +
    
    scale_fill_manual(values  = genotype_color_bar,   name = "Genotype") +
    scale_color_manual(values = genotype_color_ticks, name = "Genotype") +
  
    labs(title = paste0("Average RMSF for ", region_name,
                        " (Residues ", start_res, "-", end_res, ")"),
         x = "Residue", y = "RMSF (Å)") +
    
    theme_pubr(base_size = 18) +
    theme(text           = element_text(size = 18, face = "bold"),
          plot.title     = element_text(hjust = 0.5),
          axis.text.x    = element_text(angle = 45, hjust = 1),
          legend.position = "top", legend.box.just = "center") +
    guides(fill = guide_legend(nrow = 1))
  
  filename_avg <- paste0("plots/bar_plots/rmsf_bar_",
                         gsub(" ", "_", tolower(region_name)), ".png")
  ggsave(filename_avg, width = 20, height = 8, dpi = 500)
}

# Nucleotide Binding Site (NBS)--------------------------------------
# Reuses df_all and df_avg already in memory
# Regions for the Nucleotide binding site
nbs_regions <- list(
  "P Loop"     = c(179, 183),
  "Purine Loop"= c(126, 131),
  "SH1"        = c(674, 685),
  "S1"         = c(232, 244)
)

# NBS colors pulled from domain_colors
nbs_bg_colors <- domain_colors[c("P Loop", "Purine Loop", "SH1", "S1", "ADP.Pi")]

# create df for all regions and residues in NBS 
df_nbs <- map_dfr(names(nbs_regions), function(rname) {
  r <- nbs_regions[[rname]]
  
  df_all %>%
    filter(residue >= r[1] & residue <= r[2]) %>%
    mutate(region = rname)
}) %>%
  distinct(genotype, sim_id, residue, rmsf, .keep_all = TRUE)

# creates df for the average rmsf for each residue 
df_nbs_avg <- df_nbs %>%
  group_by(region, genotype, residue) %>%
  summarise(
    mean_rmsf = mean(rmsf),
    se_rmsf   = sd(rmsf) / sqrt(n()),
    .groups   = "drop"
  )

# creates df for all the stats since it breaks with stat_compare_means()
df_nbs_pval <- df_nbs %>%
  group_by(region, residue) %>%
  summarise(
    # try catch will catch errors in t-test for missing data
    pval = tryCatch(
      t.test(rmsf[genotype == "WT"], rmsf[genotype == "D239N"])$p.value,
      error = function(e) NA_real_
    ),
    .groups = "drop"
  ) %>%
  mutate(stars = case_when(
    pval < 0.001 ~ "***",
    pval < 0.01  ~ "**",
    pval < 0.05  ~ "*",
    TRUE          ~ ""
  ))

# Orders residues by region first thn resdue
residue_order <- df_nbs_avg %>%
  arrange(match(region, names(nbs_regions)), residue) %>%
  pull(residue) %>%
  unique()

# Maps residues to a sequential integer x position (1, 2, 3)
x_map <- setNames(seq_along(residue_order), as.character(residue_order))

# Apply x_pos to all three dataframes so they share the same x coordinate system
# All three need it because they are all used as separate layers in the plot
df_nbs_avg  <- df_nbs_avg  %>% mutate(x_pos = x_map[as.character(residue)])
df_nbs      <- df_nbs      %>% mutate(x_pos = x_map[as.character(residue)])
df_nbs_pval <- df_nbs_pval %>% mutate(x_pos = x_map[as.character(residue)])

# Creates rectangle coordinates for each region
bg_df <- df_nbs_avg %>%
  select(region, x_pos) %>%
  distinct() %>%
  group_by(region) %>%
  summarise(
    xmin  = min(x_pos) - 0.5,
    xmax  = max(x_pos) + 0.5,
    x_mid = (min(x_pos) + max(x_pos)) / 2,
    .groups = "drop"
  ) %>%
  mutate(bg_fill = nbs_bg_colors[region])

# Calculate y-axis limits from the tallest bar including its error bar
y_max   <- max(df_nbs_avg$mean_rmsf + df_nbs_avg$se_rmsf, na.rm = TRUE)
y_range <- y_max * 1.20
y_stars <- y_max * 1.05

# Filter to only significant residues for star annotations
# y_pos assigns the fixed star height calculated above
df_nbs_stars <- df_nbs_pval %>%
  filter(stars != "") %>%
  mutate(y_pos = y_stars)

# Bar plot of all residues in the NBS
ggplot() +
  # Region shading
  annotate("rect",
           xmin  = bg_df$xmin, xmax = bg_df$xmax,
           ymin  = -Inf,        ymax = Inf,
           fill  = scales::alpha(bg_df$bg_fill, 0.4)) +
  
  geom_text(data = bg_df,
            aes(x = xmin + 0.1, y = y_range * 0.98, label = region),
            inherit.aes = FALSE,
            size = 3.5, fontface = "bold", hjust = 0, vjust = 1,
            color = "black") +
  
  geom_col(data = df_nbs_avg,
           aes(x = x_pos, y = mean_rmsf, fill = genotype),
           position = position_dodge(width = 0.8), width = 0.70,
           colour = "black", linewidth = 0.3) +
  
  geom_errorbar(data = df_nbs_avg,
                aes(x     = x_pos,
                    ymin  = mean_rmsf - se_rmsf,
                    ymax  = mean_rmsf + se_rmsf,
                    group = genotype),
                position  = position_dodge(width = 0.9),
                width     = 0.2, linewidth = 0.5) +
  
  geom_point(data = df_nbs,
             aes(x = x_pos, y = rmsf,
                 group = genotype, fill = genotype, color = genotype),
             position    = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.9),
             size        = 1.5, shape = 21, stroke = 1,
             show.legend = FALSE) +
  
  geom_vline(
    xintercept = seq(1.5, length(residue_order) - 0.5, by = 1),
    color = "grey80", linewidth = 0.4, linetype = "dashed") +
  geom_text(data        = df_nbs_stars,
            aes(x = x_pos, y = y_pos, label = stars),
            inherit.aes = FALSE,
            size = 5, vjust = 0) +
  
  scale_x_continuous(
    breaks = seq_along(residue_order),
    labels = residue_order
  ) +
  
  scale_fill_manual(values  = genotype_color_bar, name = NULL) +
  scale_color_manual(values = genotype_color_ticks, name = NULL) +
  
  labs(title = "Nucleotide Binding Site RMSF",
       x = "Residue", y = "RMSF (Å)") +
  theme_pubr() +
  theme(text           = element_text(size = 18, face = "bold"),
        plot.title     = element_text(hjust = 0.5),
        axis.text.x    = element_text(angle = 45, hjust = 1),
        legend.position = "top", legend.box.just = "center") 

ggsave("plots/bar_plots/rmsf_nucleotide_binding_site_wt_d239n.png",
       width = 20, height = 8, dpi = 500)

# Nucleotide Binding Site — significant residues only (p < 0.05) ---------------
# Filter to only significant residues 
sig_residues <- df_nbs_pval %>%
  filter(pval < 0.05) %>%
  pull(residue)

df_nbs_avg_sig   <- df_nbs_avg  %>% filter(residue %in% sig_residues)
df_nbs_sig       <- df_nbs      %>% filter(residue %in% sig_residues)
df_nbs_stars_sig <- df_nbs_pval %>%
  filter(residue %in% sig_residues, stars != "")

residue_order_sig <- df_nbs_avg_sig %>%
  arrange(match(region, names(nbs_regions)), residue) %>%
  pull(residue) %>%
  unique()

x_map_sig <- setNames(seq_along(residue_order_sig), as.character(residue_order_sig))

df_nbs_avg_sig   <- df_nbs_avg_sig   %>% mutate(x_pos = x_map_sig[as.character(residue)])
df_nbs_sig       <- df_nbs_sig       %>% mutate(x_pos = x_map_sig[as.character(residue)])
df_nbs_stars_sig <- df_nbs_stars_sig %>% mutate(x_pos = x_map_sig[as.character(residue)])

bg_df_sig <- df_nbs_avg_sig %>%
  select(region, x_pos) %>%
  distinct() %>%
  group_by(region) %>%
  summarise(
    xmin  = min(x_pos) - 0.5,
    xmax  = max(x_pos) + 0.5,
    x_mid = (min(x_pos) + max(x_pos)) / 2,
    .groups = "drop"
  ) %>%
  mutate(bg_fill = nbs_bg_colors[region])

y_max_sig   <- max(df_nbs_avg_sig$mean_rmsf + df_nbs_avg_sig$se_rmsf, na.rm = TRUE)
y_range_sig <- y_max_sig * 1.20
y_stars_sig <- y_max_sig * 1.05

df_nbs_stars_sig <- df_nbs_stars_sig %>% mutate(y_pos = y_stars_sig)

# Bar plot of all significant residues in the NBS
ggplot() +
  annotate("rect",
           xmin  = bg_df_sig$xmin, xmax = bg_df_sig$xmax,
           ymin  = -Inf,            ymax = Inf,
           fill  = scales::alpha(bg_df_sig$bg_fill, 0.3)) +
  
  geom_text(data        = bg_df_sig,
            aes(x = xmin + 0.1, y = y_range_sig * 0.97, label = region),
            inherit.aes = FALSE,
            size = 8, fontface = "bold", hjust = 0, vjust = 1,
            color = "black") +
  
  geom_col(data     = df_nbs_avg_sig,
           aes(x = x_pos, y = mean_rmsf, fill = genotype),
           position = position_dodge(width = 0.8), width = 0.80,
           colour = "black", linewidth = 0.3) +
  
  geom_errorbar(data = df_nbs_avg_sig,
                aes(x     = x_pos,
                    ymin  = mean_rmsf - se_rmsf,
                    ymax  = mean_rmsf + se_rmsf,
                    group = genotype),
                position  = position_dodge(width = 0.9),
                width     = 0.2, linewidth = 0.5) +
  
  geom_point(data        = df_nbs_sig,
             aes(x = x_pos, y = rmsf,
                 group = genotype, fill = genotype, color = genotype),
             position    = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.9),
             size        = 1.5, shape = 21, stroke = 1,
             show.legend = FALSE) +
  
  geom_vline(
    xintercept = seq(1.5, length(residue_order_sig) - 0.5, by = 1),
    color = "darkgrey", linewidth = 0.4, linetype = "dashed") +
  
  geom_text(data        = df_nbs_stars_sig,
            aes(x = x_pos, y = y_pos, label = stars),
            inherit.aes = FALSE,
            size = 8, vjust = 0) +
  
  scale_x_continuous(
    breaks = seq_along(residue_order_sig),
    labels = residue_order_sig
  ) +

  scale_fill_manual(values  = genotype_color_bar, name = NULL) +
  scale_color_manual(values = genotype_color_ticks, name = NULL) +
  
  labs(title = "Nucletide Binding Site RMSF (p-val < 0.05)",
       x     = "Residue",
       y     = "RMSF (Å)") +
  theme_pubr(base_size = 25) +
  theme(text           = element_text(size = 18, face = "bold"),
        plot.title     = element_text(hjust = 0.5),
        axis.text.x    = element_text(angle = 45, hjust = 1),
        legend.position = "top", legend.box.just = "center") 
  
ggsave("plots/bar_plots/rmsf_nucleotide_binding_site_significant_wt_d239n.png",
       width = 25, height = 15, dpi = 500)

print("Analysis complete! Plots saved in 'plots/' directory.")

# Standalone ADP.Pi Bar Plot ---------------------------------------------------
adppi_range <- regions[["ADP.Pi"]]
start_res   <- adppi_range[1]
end_res      <- adppi_range[2]

df_adppi     <- df_all %>% filter(residue >= start_res & residue <= end_res)
df_adppi_avg <- df_avg %>% filter(residue >= start_res & residue <= end_res)

adppi_labels <- c("852" = "ADP", "853" = "Pi")

df_adppi     <- df_adppi     %>% 
  mutate(residue_label = factor(adppi_labels[as.character(residue)], levels = c("ADP", "Pi")))
df_adppi_avg <- df_adppi_avg %>% 
  mutate(residue_label = factor(adppi_labels[as.character(residue)], levels = c("ADP", "Pi")))

# All sims bar plot
ggplot() +
  geom_col(data = df_adppi,
           aes(x = residue_label, y = rmsf,
               fill = sim_id),
           position = position_dodge(width = 0.8), width = 0.80,
           colour = "black", linewidth = 0.4) +
  
  scale_fill_manual(values = sim_colors, name = "Simulation") +
  
  labs(title = "RMSF for ADP.Pi",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr() +
  theme(text            = element_text(size = 18, face = "bold"),
        plot.title      = element_text(hjust = 0.5),
        axis.text.x     = element_text(angle = 45, hjust = 1),
        legend.position = "top", legend.box.just = "center") +
  guides(fill = guide_legend(nrow = 1, ncol = 6, byrow = TRUE))

ggsave("plots/bar_plots/rmsf_bar_wt_d239n_adp.pi.png",
       width = 16, height = 8, dpi = 500)

# Genotype averaged bar plot with SEM, sim ticks, and t-test
ggplot() +
  geom_col(data     = df_adppi_avg,
           aes(x = residue_label, y = mean_rmsf, fill = genotype),
           position = position_dodge(width = 0.9), width = 0.80,
           colour = "black", linewidth = 0.4) +
  
  geom_errorbar(data = df_adppi_avg,
                aes(x     = residue_label,
                    ymin  = mean_rmsf - se_rmsf,
                    ymax  = mean_rmsf + se_rmsf,
                    group = genotype),
                position  = position_dodge(width = 0.9),
                width     = 0.2) +
  
  geom_point(data = df_adppi,
             aes(x = residue_label, y = rmsf,
                 group = genotype, fill = genotype, color = genotype),
             position    = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.9),
             size        = 1.5, shape = 21, stroke = 1,
             show.legend = FALSE) +
  
  stat_compare_means(
    data        = df_adppi,
    aes(x = residue_label, y = rmsf, group = genotype),
    method      = "t.test",
    label       = "p.signif",
    size        = 10,
    label.y     = max(df_adppi_avg$mean_rmsf + df_adppi_avg$se_rmsf) * 1.07,
    inherit.aes = FALSE,
    hide.ns     = TRUE) +
  
  scale_fill_manual(values  = genotype_color_bar,   name = "Genotype") +
  scale_color_manual(values = genotype_color_ticks, name = "Genotype") +
  
  labs(title = "Average RMSF for ADP.Pi",
       x = "Residue", y = "RMSF (Å)") +
  
  theme_pubr(base_size = 18) +
  theme(text            = element_text(size = 18, face = "bold"),
        plot.title      = element_text(hjust = 0.5),
        axis.text.x     = element_text(angle = 0, hjust = 0.5),
        legend.position = "top", legend.box.just = "center") +
  guides(fill = guide_legend(nrow = 1))

ggsave("plots/bar_plots/rmsf_bar_adp.pi.png",
       width = 20, height = 8, dpi = 500)