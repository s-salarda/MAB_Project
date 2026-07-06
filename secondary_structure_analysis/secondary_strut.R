# ================================================================
# Loop 2 Secondary Structure Analysis
# Genotypes: WT, D239N, K637E
# Source data: cpptraj *.sum.dat files (secondary structure per-residue
#              fractional populations across all frames)
# ================================================================

library(tidyverse)
library(ggpubr)

# ----------------------------------------------------------------
# Color scheme (consistent with other MAB scripts)
# ----------------------------------------------------------------
fill_colors <- c("WT" = "darkgrey", "D239N" = "green", "K637E" = "blue")
point_colors <- c("WT" = "black", "D239N" = "darkgreen", "K637E" = "darkblue")

genotype_levels <- c("WT", "D239N", "K637E")

# ----------------------------------------------------------------
# Structure column definitions
# ----------------------------------------------------------------
# cpptraj secstruct sum.dat columns, in raw order
structure_cols_raw <- c("Bridge", "Extended", "3-10", "Alpha", "Pi", "Turn", "Bend")

# Renamed / display labels
structure_rename <- c(
  "Bridge"   = "Parallel Beta-Sheet",
  "Extended" = "Anti-parallel Beta-sheet",
  "3-10"     = "3-10 Helix",
  "Alpha"    = "Alpha helix",
  "Pi"       = "Pi (3-14) helix"
)

desired_order <- c(
  "None", "Parallel Beta-Sheet", "Anti-parallel Beta-sheet",
  "3-10 Helix", "Alpha helix", "Pi (3-14) helix", "Turn", "Bend"
)

structure_labels <- c(
  "None", "Parallel\nBeta-Sheet", "Anti-parallel\nBeta-sheet",
  "3-10 Helix", "Alpha helix", "Pi (3-14) helix", "Turn", "Bend"
)

# ----------------------------------------------------------------
# Significance star helper
# ----------------------------------------------------------------
sig_star <- function(p) {
  case_when(
    is.na(p)  ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE      ~ ""
  )
}

# ----------------------------------------------------------------
# 1. Read a single *.sum.dat file
# ----------------------------------------------------------------
# cpptraj secstruct sum.dat format: first row = header (col names),
# first column = residue number, remaining columns = fractional
# population of each secondary structure type for that residue.
read_sum_dat <- function(filepath, genotype_name, sim_number) {
  df <- read_table(filepath, comment = "", show_col_types = FALSE)
  
  # Standardize residue column name (header is "#Residue")
  names(df)[1] <- "Residue"
  
  df <- df %>%
    rename(!!!setNames(names(structure_rename), structure_rename)) %>%
    mutate(
      Residue = as.numeric(Residue),
      Genotype = genotype_name,
      sim = sim_number
    )
  
  df
}

# ----------------------------------------------------------------
# 2. Load all simulations for a genotype from a folder
# ----------------------------------------------------------------
load_genotype_folder <- function(folder_path, genotype_name) {
  files <- list.files(folder_path, pattern = "sum\\.dat$", full.names = TRUE)
  files <- sort(files)
  
  if (length(files) == 0) {
    stop(paste("No sum.dat files found in:", folder_path))
  }
  
  map2_dfr(files, seq_along(files), function(f, i) {
    read_sum_dat(f, genotype_name, i)
  })
}

# ----------------------------------------------------------------
# 3. Set base path and load all genotypes -> df (raw, frame/residue level)
# ----------------------------------------------------------------
base_path <- "D:/Projects/MAB_project/Loop2_Secondary_Structure/cpptraj_loop2_secondary/"

df <- bind_rows(
  load_genotype_folder(file.path(base_path, "wt"),     "WT"),
  load_genotype_folder(file.path(base_path, "d239n"),  "D239N"),
  load_genotype_folder(file.path(base_path, "k637e"),  "K637E")
) %>%
  mutate(
    Genotype = factor(Genotype, levels = genotype_levels),
    Simulation = paste0(Genotype, "_sim", sim)
  )

# ----------------------------------------------------------------
# 4. Per-simulation structure percentages -> df_sim
#    (average across all residues/frames within each simulation,
#     for each secondary structure type, expressed as %)
# ----------------------------------------------------------------
structure_cols_present <- intersect(
  unique(c(structure_rename, structure_cols_raw)),
  names(df)
)

df_sim <- df %>%
  group_by(Genotype, sim, Simulation) %>%
  summarise(
    across(all_of(structure_cols_present), ~ mean(.x, na.rm = TRUE) * 100),
    .groups = "drop"
  ) %>%
  mutate(
    None = pmax(0, 100 - rowSums(across(all_of(structure_cols_present)), na.rm = TRUE)),
    total_beta = `Parallel Beta-Sheet` + `Anti-parallel Beta-sheet`
  )

# ----------------------------------------------------------------
# 5. Genotype-level summary -> df_avg (mean ± SEM across simulations)
# ----------------------------------------------------------------
summary_cols <- c(structure_cols_present, "None", "total_beta")

df_avg <- df_sim %>%
  group_by(Genotype) %>%
  summarise(
    across(all_of(summary_cols),
           list(mean = ~ mean(.x, na.rm = TRUE),
                sem  = ~ sd(.x, na.rm = TRUE) / sqrt(sum(!is.na(.x)))),
           .names = "{.col}_{.fn}"),
    .groups = "drop"
  )

# ----------------------------------------------------------------
# Two-genotype color/level helpers for pairwise comparisons
# ----------------------------------------------------------------
pairwise_levels <- list(
  D239N = c("WT", "D239N"),
  K637E = c("WT", "K637E")
)

# ----------------------------------------------------------------
# 6. Plot: full Loop 2 secondary structure organization
#    (grouped bar plot across all structure types)
#    Three versions:
#      (a) ANOVA  - all three genotypes, star = 3-way ANOVA p-value
#      (b) D239N  - WT vs D239N only, star = Student's t-test (var.equal=TRUE)
#      (c) K637E  - WT vs K637E only, star = Student's t-test (var.equal=TRUE)
# ----------------------------------------------------------------
df_avg_long_full <- df_avg %>%
  select(Genotype, ends_with("_mean"), ends_with("_sem")) %>%
  pivot_longer(
    cols = -Genotype,
    names_to = c("Structure", "stat"),
    names_pattern = "(.+)_(mean|sem)"
  ) %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(
    Structure = factor(Structure, levels = desired_order, labels = structure_labels),
    Genotype = factor(Genotype, levels = genotype_levels)
  )

build_organization_plot <- function(df_avg_long, genotypes, fill_cols) {
  ggplot(df_avg_long, aes(x = Structure, y = mean, fill = Genotype)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9),
             color = "black", linewidth = 0.8, alpha = 0.7, width = 0.8) +
    geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem),
                  position = position_dodge(width = 0.9), width = 0.25) +
    geom_text(aes(label = sprintf("%.2f%%", mean), y = mean + sem + 1),
              position = position_dodge(width = 0.9),
              angle = 90, hjust = 0, size = 3, fontface = "bold") +
    scale_fill_manual(values = fill_cols) +
    labs(title = "Secondary Structure of Loop 2",
         x = "Secondary Structure Type",
         y = "Percent of Simulations") +
    theme_pubr() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", size = 11),
      axis.title = element_text(face = "bold", size = 14),
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      legend.title = element_text(face = "bold")
    )
}

# (a) 3-way ANOVA version
shared_ymax_full <- max(df_avg_long_full$mean + df_avg_long_full$sem, na.rm = TRUE)

df_loop2_stars_anova <- map_dfr(structure_cols_present, function(col) {
  vals <- df_sim %>% select(Genotype, value = all_of(col))
  pval <- tryCatch(summary(aov(value ~ Genotype, data = vals))[[1]][["Pr(>F)"]][1],
                   error = function(e) NA_real_)
  star <- sig_star(pval)
  
  tibble(Structure = structure_labels[match(col, desired_order)], label = star, ypos = shared_ymax_full + 3)
}) %>%
  mutate(Structure = factor(Structure, levels = structure_labels)) %>%
  filter(label != "")

p_loop2_organization_anova <- build_organization_plot(df_avg_long_full, genotype_levels, fill_colors) +
  geom_text(data = df_loop2_stars_anova, aes(x = Structure, y = ypos, label = label),
            inherit.aes = FALSE, size = 6, fontface = "bold")

# (b)/(c) Pairwise versions (WT vs D239N, WT vs K637E)
build_organization_pairwise <- function(mut_genotype) {
  geno_levels <- pairwise_levels[[mut_genotype]]
  
  df_avg_long_pair <- df_avg_long_full %>%
    filter(Genotype %in% geno_levels) %>%
    mutate(Genotype = factor(Genotype, levels = geno_levels))
  
  shared_ymax_pair <- max(df_avg_long_pair$mean + df_avg_long_pair$sem, na.rm = TRUE)
  
  df_stars <- map_dfr(structure_cols_present, function(col) {
    wt_vals  <- df_sim %>% filter(Genotype == "WT")           %>% pull(.data[[col]])
    mut_vals <- df_sim %>% filter(Genotype == mut_genotype)   %>% pull(.data[[col]])
    
    pval <- tryCatch(t.test(wt_vals, mut_vals, var.equal = TRUE)$p.value, error = function(e) NA_real_)
    star <- sig_star(pval)
    
    tibble(Structure = structure_labels[match(col, desired_order)], label = star, ypos = shared_ymax_pair + 3)
  }) %>%
    mutate(Structure = factor(Structure, levels = structure_labels)) %>%
    filter(label != "")
  
  pair_fill <- fill_colors[geno_levels]
  
  build_organization_plot(df_avg_long_pair, geno_levels, pair_fill) +
    geom_text(data = df_stars, aes(x = Structure, y = ypos, label = label),
              inherit.aes = FALSE, size = 6, fontface = "bold")
}

p_loop2_organization_d239n <- build_organization_pairwise("D239N")
p_loop2_organization_k637e <- build_organization_pairwise("K637E")

print(p_loop2_organization_anova)
print(p_loop2_organization_d239n)
print(p_loop2_organization_k637e)

plots_path <- file.path(base_path, "plots")
if (!dir.exists(plots_path)) dir.create(plots_path, recursive = TRUE)

ggsave(file.path(plots_path, "loop2_organization_anova.png"),
       plot = p_loop2_organization_anova, width = 10, height = 7, dpi = 300)
ggsave(file.path(plots_path, "loop2_organization_WTvsD239N.png"),
       plot = p_loop2_organization_d239n, width = 10, height = 7, dpi = 300)
ggsave(file.path(plots_path, "loop2_organization_WTvsK637E.png"),
       plot = p_loop2_organization_k637e, width = 10, height = 7, dpi = 300)

# ----------------------------------------------------------------
# 7. Plot: single-structure bar plots with replicate dots, SEM error
#    bars, and significance stars (no brackets)
#    Three versions per structure:
#      (a) ANOVA  - all three genotypes, star = 3-way ANOVA p-value, centered
#      (b) D239N  - WT vs D239N only, star = Student's t-test (var.equal=TRUE)
#      (c) K637E  - WT vs K637E only, star = Student's t-test (var.equal=TRUE)
# ----------------------------------------------------------------
build_base_plot <- function(df_plot, df_stats, ylabel, fill_cols, point_cols) {
  ggplot(df_plot, aes(x = Genotype, y = value)) +
    geom_bar(data = df_stats, aes(x = Genotype, y = mean, fill = Genotype),
             stat = "identity", color = "black", linewidth = 0.8,
             alpha = 0.7, width = 0.6) +
    geom_errorbar(data = df_stats,
                  aes(x = Genotype, y = mean, ymin = mean - sem, ymax = mean + sem),
                  width = 0.2) +
    geom_jitter(aes(color = Genotype), width = 0.1, size = 2, alpha = 0.9,
                show.legend = FALSE) +
    scale_fill_manual(values = fill_cols) +
    scale_color_manual(values = point_cols) +
    labs(title = NULL, x = NULL, y = ylabel) +
    theme_pubr() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", size = 12),
      axis.title.y = element_text(face = "bold", size = 14),
      legend.position = "none"
    )
}

# (a) 3-way ANOVA version
plot_structure_anova <- function(structure, ylabel) {
  df_plot <- df_sim %>%
    select(Genotype, sim, all_of(structure)) %>%
    rename(value = all_of(structure))
  
  df_stats <- df_avg %>%
    select(Genotype,
           mean = paste0(structure, "_mean"),
           sem  = paste0(structure, "_sem"))
  
  anova_pval <- summary(aov(value ~ Genotype, data = df_plot))[[1]][["Pr(>F)"]][1]
  star <- sig_star(anova_pval)
  
  ymax_plot <- max(df_stats$mean + df_stats$sem, na.rm = TRUE)
  df_star <- tibble(x = 2, y = ymax_plot + 0.5, label = star)
  
  build_base_plot(df_plot, df_stats, ylabel, fill_colors, point_colors) +
    geom_text(data = df_star, aes(x = x, y = y, label = label),
              inherit.aes = FALSE, size = 6, fontface = "bold")
}

# (b)/(c) Pairwise Student's t-test version (WT vs MUT, var.equal = TRUE)
plot_structure_pairwise <- function(structure, ylabel, mut_genotype) {
  geno_levels <- pairwise_levels[[mut_genotype]]
  
  df_plot <- df_sim %>%
    filter(Genotype %in% geno_levels) %>%
    select(Genotype, sim, all_of(structure)) %>%
    rename(value = all_of(structure)) %>%
    mutate(Genotype = factor(Genotype, levels = geno_levels))
  
  df_stats <- df_avg %>%
    filter(Genotype %in% geno_levels) %>%
    select(Genotype,
           mean = paste0(structure, "_mean"),
           sem  = paste0(structure, "_sem")) %>%
    mutate(Genotype = factor(Genotype, levels = geno_levels))
  
  wt_vals  <- df_plot %>% filter(Genotype == "WT")         %>% pull(value)
  mut_vals <- df_plot %>% filter(Genotype == mut_genotype) %>% pull(value)
  
  pval <- t.test(wt_vals, mut_vals, var.equal = TRUE)$p.value
  star <- sig_star(pval)
  
  ymax_plot <- max(df_stats$mean + df_stats$sem, na.rm = TRUE)
  df_star <- tibble(x = 1.5, y = ymax_plot + 0.5, label = star)
  
  pair_fill  <- fill_colors[geno_levels]
  pair_point <- point_colors[geno_levels]
  
  build_base_plot(df_plot, df_stats, ylabel, pair_fill, pair_point) +
    geom_text(data = df_star, aes(x = x, y = y, label = label),
              inherit.aes = FALSE, size = 6, fontface = "bold")
}

structures_to_plot <- list(
  list(col = "Anti-parallel Beta-sheet", ylabel = "Percent of Sim in Anti-parallel \u03b2-Sheet", name = "antiparallel_betasheet"),
  list(col = "Parallel Beta-Sheet",      ylabel = "Percent of Sim in Parallel \u03b2-Sheet",      name = "parallel_betasheet"),
  list(col = "Alpha helix",              ylabel = "Percent of Sim in Alpha Helix",              name = "alpha_helix"),
  list(col = "total_beta",               ylabel = "Percent of Total \u03b2-Sheet",               name = "total_betasheet")
)

for (s in structures_to_plot) {
  p_anova <- plot_structure_anova(s$col, s$ylabel)
  p_d239n <- plot_structure_pairwise(s$col, s$ylabel, "D239N")
  p_k637e <- plot_structure_pairwise(s$col, s$ylabel, "K637E")
  
  print(p_anova)
  print(p_d239n)
  print(p_k637e)
  
  ggsave(file.path(plots_path, paste0(s$name, "_anova.png")),
         plot = p_anova, width = 5, height = 6, dpi = 300)
  ggsave(file.path(plots_path, paste0(s$name, "_WTvsD239N.png")),
         plot = p_d239n, width = 5, height = 6, dpi = 300)
  ggsave(file.path(plots_path, paste0(s$name, "_WTvsK637E.png")),
         plot = p_k637e, width = 5, height = 6, dpi = 300)
}

# ----------------------------------------------------------------
# 8. Residue-level statistics for a given structure type
#    Computes per-residue 3-way ANOVA plus pairwise Student's t-tests
#    (WT vs D239N, WT vs K637E, var.equal = TRUE)
# ----------------------------------------------------------------
compute_residue_stats <- function(structure_col) {
  
  df_resid_sim <- df %>%
    group_by(Genotype, Simulation, Residue) %>%
    summarise(value = mean(.data[[structure_col]], na.rm = TRUE) * 100, .groups = "drop")
  
  anova_results <- df_resid_sim %>%
    group_by(Residue) %>%
    summarise(
      pval = tryCatch(
        summary(aov(value ~ Genotype, data = pick(everything())))[[1]][["Pr(>F)"]][1],
        error = function(e) NA_real_
      ),
      .groups = "drop"
    )
  
  pairwise_results <- df_resid_sim %>%
    group_by(Residue) %>%
    summarise(
      pval_d239n = tryCatch(
        t.test(value[Genotype == "WT"], value[Genotype == "D239N"], var.equal = TRUE)$p.value,
        error = function(e) NA_real_
      ),
      pval_k637e = tryCatch(
        t.test(value[Genotype == "WT"], value[Genotype == "K637E"], var.equal = TRUE)$p.value,
        error = function(e) NA_real_
      ),
      .groups = "drop"
    )
  
  full_results <- anova_results %>% left_join(pairwise_results, by = "Residue")
  
  df_stats <- df_resid_sim %>%
    group_by(Genotype, Residue) %>%
    summarise(
      mean = mean(value, na.rm = TRUE),
      sem  = sd(value, na.rm = TRUE) / sqrt(sum(!is.na(value))),
      .groups = "drop"
    )
  
  list(
    full_results = full_results,
    df_resid_sim = df_resid_sim,
    df_stats = df_stats
  )
}

# ----------------------------------------------------------------
# 9. Plot significant residues for a structure type
#    Three versions:
#      (a) ANOVA  - all three genotypes, residues significant by 3-way ANOVA,
#                    star centered above the 3-bar group
#      (b) D239N  - WT vs D239N only, residues significant by that t-test,
#                    star centered above the 2-bar group
#      (c) K637E  - WT vs K637E only, residues significant by that t-test,
#                    star centered above the 2-bar group
# ----------------------------------------------------------------
plot_significant_residues <- function(structure_col, pval_threshold = 0.05, title = NULL) {
  
  res <- compute_residue_stats(structure_col)
  safe_name <- gsub("[^A-Za-z0-9]+", "_", structure_col)
  
  # ---- (a) 3-way ANOVA version ----
  sig_anova <- res$full_results %>% filter(pval < pval_threshold) %>% pull(Residue)
  
  if (length(sig_anova) == 0) {
    message("No ANOVA-significant residues for ", structure_col)
  } else {
    df_plot <- res$df_resid_sim %>%
      filter(Residue %in% sig_anova) %>%
      mutate(Residue = factor(Residue, levels = sort(unique(Residue))))
    
    df_bar <- res$df_stats %>%
      filter(Residue %in% sig_anova) %>%
      mutate(Residue = factor(Residue, levels = sort(unique(Residue))))
    
    df_star <- res$full_results %>%
      filter(Residue %in% sig_anova) %>%
      mutate(Residue = factor(Residue, levels = sort(unique(Residue))),
             label = sig_star(pval))
    
    shared_ymax <- max(df_bar$mean + df_bar$sem, na.rm = TRUE)
    df_star <- df_star %>% mutate(ypos = shared_ymax + 1)
    
    p_anova <- ggplot(df_bar, aes(x = Residue, y = mean, fill = Genotype)) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.9),
               color = "black", linewidth = 0.6, alpha = 0.8, width = 0.8) +
      geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem),
                    position = position_dodge(width = 0.9), width = 0.25) +
      geom_point(data = df_plot,
                 aes(x = Residue, y = value, fill = Genotype),
                 position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.9),
                 shape = 21, color = "black", size = 1.5, show.legend = FALSE) +
      geom_text(data = df_star, aes(x = Residue, y = ypos, label = label),
                inherit.aes = FALSE, size = 6, fontface = "bold") +
      scale_fill_manual(values = fill_colors) +
      coord_cartesian(ylim = c(0, 17)) +
      labs(title = title %||% paste("Significant Residues for", structure_col, "(ANOVA)"),
           x = "Residue", y = "Fraction of Frames (%)") +
      theme_pubr() +
      theme(
        axis.title = element_text(face = "bold", size = 14),
        plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        legend.title = element_text(face = "bold")
      )
    
    print(p_anova)
    ggsave(file.path(plots_path, paste0("significant_residues_", safe_name, "_anova.png")),
           plot = p_anova, width = 6, height = 6, dpi = 300)
  }
  
  # ---- (b)/(c) Pairwise versions ----
  for (mut_genotype in c("D239N", "K637E")) {
    geno_levels <- pairwise_levels[[mut_genotype]]
    pval_col <- paste0("pval_", tolower(mut_genotype))
    
    sig_pair <- res$full_results %>% filter(.data[[pval_col]] < pval_threshold) %>% pull(Residue)
    
    if (length(sig_pair) == 0) {
      message("No ", mut_genotype, "-significant residues for ", structure_col)
      next
    }
    
    df_plot <- res$df_resid_sim %>%
      filter(Residue %in% sig_pair, Genotype %in% geno_levels) %>%
      mutate(Residue = factor(Residue, levels = sort(unique(Residue))),
             Genotype = factor(Genotype, levels = geno_levels))
    
    df_bar <- res$df_stats %>%
      filter(Residue %in% sig_pair, Genotype %in% geno_levels) %>%
      mutate(Residue = factor(Residue, levels = sort(unique(Residue))),
             Genotype = factor(Genotype, levels = geno_levels))
    
    df_star <- res$full_results %>%
      filter(Residue %in% sig_pair) %>%
      mutate(Residue = factor(Residue, levels = sort(unique(Residue))),
             label = sig_star(.data[[pval_col]]))
    
    shared_ymax <- max(df_bar$mean + df_bar$sem, na.rm = TRUE)
    df_star <- df_star %>% mutate(ypos = shared_ymax + 1)
    
    pair_fill <- fill_colors[geno_levels]
    
    p_pair <- ggplot(df_bar, aes(x = Residue, y = mean, fill = Genotype)) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.9),
               color = "black", linewidth = 0.6, alpha = 0.8, width = 0.8) +
      geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem),
                    position = position_dodge(width = 0.9), width = 0.25) +
      geom_point(data = df_plot,
                 aes(x = Residue, y = value, fill = Genotype),
                 position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.9),
                 shape = 21, color = "black", size = 1.5, show.legend = FALSE) +
      geom_text(data = df_star, aes(x = Residue, y = ypos, label = label),
                inherit.aes = FALSE, size = 6, fontface = "bold") +
      scale_fill_manual(values = pair_fill) +
      coord_cartesian(ylim = c(0, 17)) +
      labs(title = (title %||% paste("Significant Residues for", structure_col)) %>%
             paste0(" (WT vs ", mut_genotype, ")"),
           x = "Residue", y = "Fraction of Frames (%)") +
      theme_pubr() +
      theme(
        axis.title = element_text(face = "bold", size = 14),
        plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        legend.title = element_text(face = "bold")
      )
    
    print(p_pair)
    ggsave(file.path(plots_path, paste0("significant_residues_", safe_name, "_WTvs", mut_genotype, ".png")),
           plot = p_pair, width = 6, height = 6, dpi = 300)
  }
  
  res$full_results
}

# Run for anti-parallel beta-sheet (analogous to "Extended" in python script)
sig_residues_antiparallel <- plot_significant_residues(
  structure_col = "Anti-parallel Beta-sheet",
  pval_threshold = 0.05,
  title = "Anti-parallel Beta-Sheets in Loop 2"
)

# ----------------------------------------------------------------
# 10. Save outputs
# ----------------------------------------------------------------
results_path <- file.path(base_path, "results")
if (!dir.exists(results_path)) dir.create(results_path, recursive = TRUE)

write_csv(df,      file.path(results_path, "loop2_secondary_structure_raw.csv"))
write_csv(df_sim,  file.path(results_path, "loop2_secondary_structure_per_sim.csv"))
write_csv(df_avg,  file.path(results_path, "loop2_secondary_structure_summary.csv"))