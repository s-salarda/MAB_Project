# Coulombic Energy Analysis ----------------------------------------------------
library(tidyverse)
library(dplyr)
library(purrr)
library(ggplot2)
library(ggpubr)

# Colors -----------------------------------------------------------------------
genotype_colors <- c(
  "WT"  = "darkgrey",
  "MUT" = "green"
)

genotype_colors_ticks <- c(
  "WT"  = "black",
  "MUT" = "darkgreen"
)

sim_colors <- c(
  "WT1"  = "black",     "WT2"  = "darkgray",   "WT3"  = "lightgray",
  "MUT1" = "darkgreen", "MUT2" = "green",       "MUT3" = "lightgreen"
)

section_colors <- c(
  "P Loop"      = "#E8E8FF",
  "Purine Loop" = "#E8E8FF",
  "SH1"         = "#9090CF",
  "Switch I"          = "#FF9999",
  "S2"          = "#FFB899",
  "Loop 2"      = "#E07050",
  "Loop 3"      = "#E07050",
  "Upper 50kDa" = "#A8E4EC",
  "Lower 50kDa" = "#90C4E8",
  "ADP.Pi"      = "#90E0A0",
  "Loop 4"      = "#E07050",
  "N-terminus"  = "#f2e9dd"
)

# User Settings ----------------------------------------------------------------
BASE_DIR <- "D:/Projects/MAB_project/Columbic"
WT_DIR   <- file.path(BASE_DIR, "WT")
MUT_DIR  <- file.path(BASE_DIR, "d239n")
SAVE_DIR <- file.path(BASE_DIR, "plots")

SECTION_RENAME <- c(
  "SH1"         = "SH1",
  "S1"          = "Switch I",
  "P-loop"      = "P Loop",
  "P-Loop"      = "P Loop",
  "Purine"      = "Purine Loop",
  "Loop_2"      = "Loop 2",
  "Loop_3"      = "Loop 3",
  "Loop_4"      = "Loop 4",
  "Upper_50kDa" = "Upper 50kDa",
  "Lower_50kDa" = "Lower 50kDa"
)

# Create output dir ------------------------------------------------------------
dir.create(SAVE_DIR, recursive = TRUE, showWarnings = FALSE)

# ── 1. Parse + collect all .dat files ----------------------------------------
parse_dat_file <- function(path, state, sim, section, resid) {
  tryCatch({
    header <- read_lines(path, n_max = 1) |>
      str_remove("^#") |>
      str_trim() |>
      str_split("\\s+") |>
      unlist() |>
      make.names()

    read_table(path, skip = 1, col_names = header,
               col_types = cols(.default = "d"),
               comment = "") |>
      select(frame = Frame, electrostatic = mask.elec.) |>
      mutate(state   = state,
             sim     = sim,
             section = section,
             resid   = as.integer(resid))

  }, error = function(e) {
    message("Failed to parse: ", path, " — ", e$message)
    NULL
  })
}

collect_all_data <- function() {
  state_dirs <- list(
    list(dir = WT_DIR,  state = "WT"),
    list(dir = MUT_DIR, state = "MUT")
  )

  map_dfr(state_dirs, function(state_info) {
    sim_dirs <- list.dirs(state_info$dir, full.names = TRUE, recursive = FALSE)

    map_dfr(sim_dirs, function(sim_dir) {
      simname <- basename(sim_dir)
      files   <- list.files(sim_dir, pattern = "\\.dat$", full.names = TRUE)

      map_dfr(files, function(path) {
        m <- str_match(basename(path), "^(.+)_([0-9]+)\\.dat$")
        if (is.na(m[1])) return(NULL)

        section_raw <- m[2]
        resid       <- m[3]
        section     <- ifelse(section_raw %in% names(SECTION_RENAME),
                              SECTION_RENAME[section_raw], section_raw)

        parse_dat_file(path, state_info$state, simname, section, resid)
      })
    })
  }) |>
    mutate(state = factor(state, levels = c("WT", "MUT")))
}

# ── 2. Build per-simulation average df ---------------------------------------
build_sim_avg <- function(df) {
  df |>
    group_by(state, sim, section, resid) |>
    summarise(avg_elec = mean(electrostatic), .groups = "drop") |>
    mutate(
      state  = factor(state, levels = c("WT", "MUT")),
      sim_id = paste0(toupper(state), str_extract(sim, "[0-9]+"))
    )
}

# ── 3. Build summary df (mean ± SEM per residue × state) ---------------------
build_summary <- function(df_sim) {
  df_sim |>
    mutate(state = as.character(state)) |>
    group_by(state, section, resid) |>
    summarise(
      mean_elec = mean(avg_elec),
      sd_elec   = sd(avg_elec),
      se_elec   = sd(avg_elec) / sqrt(n()),
      .groups   = "drop"
    ) |>
    mutate(state = factor(state, levels = c("WT", "MUT")))
}

# ── 4. Compute significant residues ------------------------------------------
compute_significant_residues <- function(df_sim, pval_thresh = 0.05) {
  df_sim |>
    mutate(state = as.character(state)) |>
    group_by(section, resid) |>
    summarise(
      p_val = tryCatch(
        t.test(avg_elec[state == "WT"],
               avg_elec[state == "MUT"],
               var.equal = TRUE)$p.value,  # Student's t-test
        error = function(e) NA_real_
      ),
      .groups = "drop"
    ) |>
    filter(!is.na(p_val) & p_val < pval_thresh)
}

p_to_stars <- function(p) {
  case_when(
    is.na(p)  ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE       ~ ""
  )
}

# ── 5. Significant residues bar plot -----------------------------------------
plot_significant_barplot <- function(df_sim, pval_thresh = 0.05,
                                     title = "Electrostatic Energy of Significant Residues") {

  # force resid to character throughout to avoid continuous scale conflicts
  df_sim <- df_sim |> mutate(resid = as.character(resid))

  sig <- compute_significant_residues(df_sim, pval_thresh) |>
    arrange(section, resid) |>
    mutate(resid = as.character(resid))

  if (nrow(sig) == 0) {
    message("No significant residues found.")
    return(invisible(NULL))
  }

  resid_levels <- unique(sig$resid)

  df_sub <- df_sim |>
    mutate(state = as.character(state)) |>
    filter(resid %in% resid_levels) |>
    mutate(resid = factor(resid, levels = resid_levels),
           state = factor(state, levels = c("WT", "MUT")))

  df_avg_sub <- build_summary(df_sub) |>
    mutate(resid = factor(as.character(resid), levels = resid_levels),
           state = factor(state, levels = c("WT", "MUT")))

  # Manual Student's t-test p-values per resid --------------------------------
  pvals <- df_sub |>
    mutate(state = as.character(state)) |>
    group_by(resid) |>
    summarise(
      p_val = tryCatch(
        t.test(avg_elec[state == "WT"],
               avg_elec[state == "MUT"],
               var.equal = TRUE)$p.value,
        error = function(e) NA_real_
      ),
      .groups = "drop"
    ) |>
    mutate(
      p_signif = case_when(
        is.na(p_val)  ~ NA_character_,
        p_val < 0.001 ~ "***",
        p_val < 0.01  ~ "**",
        p_val < 0.05  ~ "*",
        TRUE           ~ "ns"
      )
    ) |>
    filter(!is.na(p_signif) & p_signif != "ns")

  y_tops <- df_avg_sub |>
    group_by(resid) |>
    summarise(y_top = max(mean_elec + se_elec), .groups = "drop")

  pvals <- left_join(pvals, y_tops, by = "resid")

  # Background shading per section — use integer positions after scale is set -
  bg_df <- sig |>
    mutate(x_pos = as.integer(factor(resid, levels = resid_levels))) |>
    group_by(section) |>
    summarise(
      xmin  = min(x_pos) - 0.5,
      xmax  = max(x_pos) + 0.5,
      x_mid = (min(x_pos) + max(x_pos)) / 2,
      .groups = "drop"
    ) |>
    mutate(bg_fill = scales::alpha(
      coalesce(section_colors[section], "#F0F0F0"), 0.35
    ))

  y_max <- max(df_avg_sub$mean_elec + df_avg_sub$se_elec, na.rm = TRUE) * 1.35
  y_min <- min(df_avg_sub$mean_elec - df_avg_sub$se_elec, na.rm = TRUE) * 1.3

  ggplot() +
    # discrete scale declared first so annotate numeric positions don't hijack it
    scale_x_discrete() +
    annotate("rect",
             xmin = bg_df$xmin, xmax = bg_df$xmax,
             ymin = -Inf,        ymax = Inf,
             fill = bg_df$bg_fill) +
    annotate("text",
             x     = bg_df$x_mid,
             y     = Inf,
             label = bg_df$section,
             vjust = 1.5, size = 4, fontface = "bold") +
    geom_col(data = df_avg_sub,
             aes(x = resid, y = mean_elec, fill = state),
             position = position_dodge(0.7), width = 0.6,
             colour = "black", linewidth = 0.3) +
    geom_errorbar(data = df_avg_sub,
                  aes(x     = resid,
                      ymin  = mean_elec - se_elec,
                      ymax  = mean_elec + se_elec,
                      group = state),
                  position = position_dodge(0.7), width = 0.2) +
    geom_point(data = df_sub,
               aes(x     = resid,
                   y     = avg_elec,
                   group = state,
                   color = state),
               position    = position_jitterdodge(jitter.width = 0.05,
                                                  dodge.width  = 0.7),
               size        = 2, shape = 16, show.legend = FALSE) +
    geom_text(data        = pvals,
              aes(x = resid, y = y_max * .9, label = p_signif),
              size        = 6,
              fontface    = "bold",
              inherit.aes = FALSE) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.8) +
    scale_fill_manual(values  = genotype_colors, name = NULL) +
    scale_color_manual(values = genotype_colors_ticks, name = NULL) +
    labs(title = title,
         x     = "Residue",
         y     = "Electrostatic Energy (kcal/mol)") +
    coord_cartesian(ylim = c(y_min, y_max), clip = "off") +
    theme_pubr(base_size = 14) +
    theme(text            = element_text(size = 14, face = "bold"),
          plot.title      = element_text(hjust = 0.5),
          axis.text.x     = element_text(angle = 0, hjust = 0.5),
          legend.position = "top",
          plot.margin     = margin(t = 20, r = 10, b = 10, l = 10))
}

# ── 6. Heatmap (tile plot) ----------------------------------------------------
plot_heatmap <- function(df_sim) {
  df_avg <- build_summary(df_sim)

  sig <- compute_significant_residues(df_sim) |>
    mutate(stars = p_to_stars(p_val),
           state = factor("MUT", levels = c("WT", "MUT")))

  for (sec in sort(unique(df_avg$section))) {
    sub <- df_avg |>
      filter(section == sec) |>
      mutate(state = factor(state, levels = c("WT", "MUT")),
             resid = factor(resid))

    sub_stars <- sig |>
      filter(section == sec) |>
      mutate(resid = factor(resid))

    p <- ggplot(sub, aes(x = state, y = resid, fill = mean_elec)) +
      geom_tile(color = "gray", linewidth = 0.4) +
      geom_text(data = sub_stars,
                aes(x = state, y = resid, label = stars),
                inherit.aes = FALSE,
                size = 6, fontface = "bold", color = "black") +
      scale_fill_gradient2(low      = "darkgrey",
                           mid      = "white",
                           high     = "darkgreen",
                           midpoint = 0,
                           name     = "kcal/mol") +
      labs(title = paste0("Electrostatic Energy of ", sec),
           x = "State", y = "Residue") +
      theme_pubr(base_size = 14) +
      theme(text            = element_text(size = 14, face = "bold"),
            plot.title      = element_text(hjust = 0.5),
            legend.position = "right")

    print(p)
    ggsave(file.path(SAVE_DIR, paste0("heatmap_", gsub(" ", "_", sec), ".png")),
           plot = p, width = 6, height = 8, dpi = 500)
  }
}

# ── 7. Energy over frames (mean ± SEM across sims) ---------------------------
plot_energy_over_frames <- function(df, df_sim) {
  sig      <- compute_significant_residues(df_sim)
  residues <- sig$resid

  for (res in residues) {
    sec  <- sig |> filter(resid == res) |> pull(section) |> first()
    df_r <- df  |> filter(resid == res)

    frame_stats <- df_r |>
      group_by(state, frame) |>
      summarise(mean_e = mean(electrostatic),
                sem_e  = sd(electrostatic) / sqrt(n()),
                .groups = "drop")

    p <- ggplot(frame_stats,
                aes(x = frame, y = mean_e, color = state, fill = state)) +
      geom_line(linewidth = 1.5) +
      geom_ribbon(aes(ymin = mean_e - sem_e, ymax = mean_e + sem_e),
                  alpha = 0.3, color = NA) +
      scale_color_manual(values = genotype_colors, name = NULL) +
      scale_fill_manual(values  = genotype_colors, name = NULL) +
      labs(title = paste0("Residue ", res, " (", sec, ")"),
           x = "Frame", y = "Electrostatic Energy (kcal/mol)") +
      theme_pubr(base_size = 14) +
      theme(text            = element_text(size = 14, face = "bold"),
            plot.title      = element_text(hjust = 0.5),
            legend.position = "top")

    print(p)
    ggsave(file.path(SAVE_DIR, paste0("energy_over_frames_", res, ".png")),
           plot = p, width = 12, height = 6, dpi = 500)
  }
}

# ── 8. WT vs MUT avg (two-panel per residue) ----------------------------------
plot_energy_avg <- function(df, df_sim) {
  sig      <- compute_significant_residues(df_sim)
  residues <- sig$resid

  for (res in residues) {
    sec  <- sig |> filter(resid == res) |> pull(section) |> first()
    df_r <- df  |> filter(resid == res)

    panel_stats <- map_dfr(c("WT", "MUT"), function(st) {
      df_r |>
        filter(as.character(state) == st) |>
        group_by(sim, frame) |>
        summarise(e = mean(electrostatic), .groups = "drop") |>
        group_by(frame) |>
        summarise(mean_e = mean(e),
                  sem_e  = sd(e) / sqrt(n()),
                  .groups = "drop") |>
        mutate(state        = st,
               overall_mean = mean(mean_e))
    }) |>
      mutate(state = factor(state, levels = c("WT", "MUT")))

    y_min <- min(panel_stats$mean_e - panel_stats$sem_e)
    y_max <- max(panel_stats$mean_e + panel_stats$sem_e)
    y_pad <- (y_max - y_min) * 0.10

    p <- ggplot(panel_stats,
                aes(x = frame, y = mean_e, color = state, fill = state)) +
      geom_line(linewidth = 1.5) +
      geom_ribbon(aes(ymin = mean_e - sem_e, ymax = mean_e + sem_e),
                  alpha = 0.3, color = NA) +
      geom_hline(aes(yintercept = overall_mean, color = state),
                 linetype = "dashed", linewidth = 1.0) +
      facet_wrap(~state, ncol = 2) +
      scale_color_manual(values = genotype_colors, name = NULL) +
      scale_fill_manual(values  = genotype_colors, name = NULL) +
      coord_cartesian(ylim = c(y_min - y_pad, y_max + y_pad)) +
      labs(title = paste0("Electrostatic Energy of Residue ", res, " (", sec, ")"),
           x = "Frame", y = "Electrostatic Energy (kcal/mol)") +
      theme_pubr(base_size = 14) +
      theme(text            = element_text(size = 14, face = "bold"),
            plot.title      = element_text(hjust = 0.5),
            legend.position = "none",
            strip.text      = element_text(size = 14, face = "bold"))

    print(p)
    ggsave(file.path(SAVE_DIR, paste0("energy_avg_", res, ".png")),
           plot = p, width = 18, height = 8, dpi = 500)
  }
}

# ── 9. Each sim (two-panel per residue) ---------------------------------------
plot_energy_each_sim <- function(df, df_sim) {
  sig      <- compute_significant_residues(df_sim)
  residues <- sig$resid

  for (res in residues) {
    sec  <- sig |> filter(resid == res) |> pull(section) |> first()
    df_r <- df  |> filter(resid == res)

    sim_frame <- df_r |>
      group_by(state, sim, frame) |>
      summarise(e = mean(electrostatic), .groups = "drop") |>
      mutate(sim_id = paste0(toupper(state), str_extract(sim, "[0-9]+")),
             state  = factor(state, levels = c("WT", "MUT")))

    band <- sim_frame |>
      group_by(state, frame) |>
      summarise(mean_e = mean(e),
                sem_e  = sd(e) / sqrt(n()),
                .groups = "drop")

    y_min <- min(sim_frame$e)
    y_max <- max(sim_frame$e)
    y_pad <- (y_max - y_min) * 0.10

    p <- ggplot() +
      geom_line(data = sim_frame,
                aes(x = frame, y = e, color = sim_id, group = sim_id),
                linewidth = 1.2) +
      geom_ribbon(data = band,
                  aes(x     = frame,
                      ymin  = mean_e - sem_e,
                      ymax  = mean_e + sem_e,
                      group = state),
                  fill = "black", alpha = 0.15, color = NA) +
      facet_wrap(~state, ncol = 2,
                 labeller = labeller(state = c("WT"  = "WT Simulations",
                                               "MUT" = "MUT Simulations"))) +
      scale_color_manual(values = sim_colors, name = "Simulation") +
      coord_cartesian(ylim = c(y_min - y_pad, y_max + y_pad)) +
      labs(title = paste0("Electrostatic Energy of Residue ", res, " (", sec, ")"),
           x = "Frame", y = "Electrostatic Energy (kcal/mol)") +
      theme_pubr(base_size = 14) +
      theme(text            = element_text(size = 14, face = "bold"),
            plot.title      = element_text(hjust = 0.5),
            legend.position = "top",
            strip.text      = element_text(size = 14, face = "bold"))

    print(p)
    ggsave(file.path(SAVE_DIR, paste0("energy_sims_", res, ".png")),
           plot = p, width = 18, height = 8, dpi = 500)
  }
}

# ── Run -----------------------------------------------------------------------
message("Loading data...")
df     <- collect_all_data()
df_sim <- build_sim_avg(df)

message("Loaded ", nrow(df), " rows | ",
        n_distinct(df$section), " sections | ",
        n_distinct(df$resid),   " residues | ",
        n_distinct(df$sim),     " simulations")
# 
# # Heatmaps
# message("Plotting heatmaps...")
# plot_heatmap(df_sim)

# Significant residues — specific residues
message("Plotting significant residue bar plot...")
df_sim_filtered <- df_sim |>
  filter(resid %in% c(227, 235, 239, 316, 322, 324,
                      362, 481, 589, 647, 650, 674, 679))
p_bar <- plot_significant_barplot(
  df_sim_filtered,
  pval_thresh = 0.05,
  title       = "Electrostatic Energy of Significant Residues"
)
print(p_bar)
ggsave(file.path(SAVE_DIR, "barplot_significant_residues.png"),
       plot = p_bar, width = 16, height = 8, dpi = 500)

# ADP section
message("Plotting ADP bar plot...")
adppi_labels <- c("852" = "ADP", "853" = "Pi")
df_sim_ADP <- df_sim |> 
  filter(resid %in% c(852, 853))  %>% 
  mutate(resid = factor(adppi_labels[as.character(resid)], levels = c("ADP", "Pi")))
  
p_adp <- plot_significant_barplot(
  df_sim_ADP,
  pval_thresh = 100,
  title       = "Electrostatic Energy of ADP.Pi"
)
print(p_adp)
ggsave(file.path(SAVE_DIR, "barplot_ADP.png"),
       plot = p_adp, width = 12, height = 8, dpi = 500)

# # Frame-level plots
# message("Plotting energy over frames...")
# plot_energy_over_frames(df, df_sim)
# 
# message("Plotting WT vs MUT avg panels...")
# plot_energy_avg(df, df_sim)
# 
# message("Plotting individual sim panels...")
# plot_energy_each_sim(df, df_sim)

message("Analysis complete! Plots saved in '", SAVE_DIR, "'")