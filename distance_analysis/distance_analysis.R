# Distance Analysis ------------------------------------------------------------
library(tidyverse)
library(dplyr)
library(purrr)
library(ggplot2)
library(ggpubr)

# Colors -----------------------------------------------------------------------
group_colors <- c(
  "WT"  = "darkgrey",
  "MUT" = "green"
)

group_colors_ticks <- c(
  "WT"  = "black",
  "MUT" = "darkgreen"
)

# File Paths -------------------------------------------------------------------
file_path <- "D:/Projects/MAB_project/Distance_Analysis"

# Distance Pairs ---------------------------------------------------------------
pair_groups <- list(
  "Nucleotide Pocket & ADP" = c(
    "ADP_SH1", "ADP_S1", "ADP_Ploop", "ADP_Purine"
  ),
  "P Loop & Phosphates" = c(
    "Ploop_Pi"
  ),
  "Switch 1 Coupling" = c(
    "S1_SH1", "S1_U50", "S1_L50", "S1_Ohelix", "S1_Relay"
  ),
  "Cleft Reshaping" = c(
    "S1_S2", "U50_L50", "HLH_L2", "L3_L2", "Ohelix_Relay"
  ),
  "Distance between Residues & 239" = c(
    "239_320", "239_321", "239_323", "239_679"
  ),
  "Phosphate Backdoor" = c(
    "237_466", "237_Pi", "466_Pi", "468_475"
  )
)

# Flat list of all pairs
pairs <- unlist(pair_groups, use.names = FALSE)

# Functions --------------------------------------------------------------------
parse_dat_file <- function(path, max_frame = 500) {
  read_table(path, comment = "#", col_names = c("Frame", "Distance"),
             col_types = "dd") |>
    filter(Frame <= max_frame) |>
    summarise(avg_distance = mean(Distance)) |>
    pull(avg_distance)
}

load_group <- function(folder, label, group, max_frame = 500) {
  files   <- list.files(folder, pattern = "\\.dat$", full.names = TRUE, recursive = TRUE)
  matches <- files[map_lgl(basename(files), \(f)
                           str_detect(tolower(f), fixed(tolower(label))) & str_detect(tolower(f), fixed(tolower(group)))
  )]
  
  if (length(matches) == 0) {
    warning("No files matched: ", label, " | ", group)
    return(NULL)
  }
  
  setNames(map_dbl(matches, parse_dat_file, max_frame = max_frame), basename(matches))
}

# Build combined long-format df ------------------------------------------------
df_all <- map_dfr(pairs, function(label) {
  wt_vals  <- load_group(file_path, label, "wt")
  mut_vals <- load_group(file_path, label, "mut")
  
  rows <- list()
  if (!is.null(wt_vals))  rows <- c(rows, list(tibble(pair = label, group = "WT",  filename = names(wt_vals),  avg_distance = unname(wt_vals))))
  if (!is.null(mut_vals)) rows <- c(rows, list(tibble(pair = label, group = "MUT", filename = names(mut_vals), avg_distance = unname(mut_vals))))
  
  if (length(rows) == 0) return(NULL)
  bind_rows(rows)
}) |>
  mutate(
    group      = factor(group, levels = c("WT", "MUT")),
    pair       = factor(pair,  levels = pairs),
    pair_group = case_when(
      pair %in% pair_groups[["Nucleotide Pocket & ADP"]] ~ "Nucleotide Pocket & ADP",
      pair %in% pair_groups[["P Loop & Phosphates"]]     ~ "P Loop & Phosphates",
      pair %in% pair_groups[["Switch 1 Coupling"]]       ~ "Switch 1 Coupling",
      pair %in% pair_groups[["Cleft Reshaping"]]         ~ "Cleft Reshaping",
      pair %in% pair_groups[["Distance between Residues & 239"]]      ~ "Distance between Residues & 239",
      pair %in% pair_groups[["Phosphate Backdoor"]]      ~ "Phosphate Backdoor"
    ),
    pair_group = factor(pair_group, levels = names(pair_groups))
  )

# Summary df (mean ± SEM per pair × group) ------------------------------------
df_avg <- df_all |>
  group_by(pair_group, pair, group) |>
  summarise(
    mean_dist = mean(avg_distance),
    sd_dist   = sd(avg_distance),
    se_dist   = sd(avg_distance) / sqrt(n()),
    .groups   = "drop"
  )

# Create output directories ----------------------------------------------------
dir.create("plots",          showWarnings = FALSE)

# Plot function ----------------------------------------------------------------
plot_group <- function(group_name) {
  
  df_sub     <- df_all |> filter(pair_group == group_name)
  df_avg_sub <- df_avg  |> filter(pair_group == group_name)
  y_max      <- max(df_avg_sub$mean_dist + df_avg_sub$se_dist) * 1.6
  
  ggplot() +
    geom_col(data = df_avg_sub,
             aes(x = pair, y = mean_dist, fill = group),
             position = position_dodge(0.6), width = 0.5) +
    geom_errorbar(data = df_avg_sub,
                  aes(x = pair,
                      ymin = mean_dist - se_dist,
                      ymax = mean_dist + se_dist,
                      group = group),
                  position = position_dodge(0.6), width = 0.15) +
    geom_point(data = df_sub,
               aes(x = pair, y = avg_distance, group = group, color = group),
               position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.6),
               size = 1.8, show.legend = FALSE) +
    stat_compare_means(data = df_sub,
                       aes(x = pair, y = avg_distance, group = group),
                       method       = "t.test",
                       method.args  = list(var.equal = TRUE),  # Student's t-test
                       label        = "p.signif",
                       hide.ns      = TRUE,
                       size         = 5,
                       label.y      = y_max * 0.92,
                       inherit.aes  = FALSE) +
    scale_fill_manual(values  = group_colors,       name = "Group") +
    scale_color_manual(values = group_colors_ticks) +
    labs(title = group_name,
         x = NULL, y = "Average Distance (Å)") +
    ylim(0, y_max) +
    theme_pubr() +
    theme(text = element_text(size = 10, face = "bold"),
          plot.title      = element_text(hjust = 0.5),
          axis.text.x     = element_text(angle = 0, hjust = 0.5),
          legend.position = "top")
}

# Generate and save plots ------------------------------------------------------
for (grp in names(pair_groups)) {
  # skip groups with no data
  if (!grp %in% unique(df_all$pair_group)) next
  
  p <- plot_group(grp)
  print(p)
  
  filename <- paste0("plots/distance/dist_",
                     gsub("[^a-zA-Z0-9]", "_", tolower(grp)), ".png")
  ggsave(filename, plot = p, width = 10, height = 6, dpi = 500)
}

print("Analysis complete! Plots saved in 'plots/distance/' directory.")
