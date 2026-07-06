# Helix Angle Analysis ---------------------------------------------------------
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

# User Settings ----------------------------------------------------------------
BASE_DIR <- "D:/Projects/MAB_project/Helix_Angles"
WT_DIR   <- file.path(BASE_DIR, "angles_wt")
MUT_DIR  <- file.path(BASE_DIR, "angles_d239n")
SAVE_DIR <- file.path(BASE_DIR, "plots")

# Domains to include -----------------------------------------------------------
DOMAINS <- c("Relay_OHelix")

# Create output dir ------------------------------------------------------------
dir.create(SAVE_DIR, recursive = TRUE, showWarnings = FALSE)

# ── 1. Parse a single .dat file ----------------------------------------------
parse_angle_file <- function(path, group, sim, domain) {
  tryCatch({
    read_table(path, comment = "#", col_names = FALSE,
               col_types = cols(.default = "d")) |>
      select(frame = X1, angle = X2) |>
      mutate(group  = group,
             sim    = sim,
             domain = domain)
  }, error = function(e) {
    message("Failed to parse: ", path, " — ", e$message)
    NULL
  })
}

# ── 2. Collect all .dat files -------------------------------------------------
collect_all_data <- function() {
  dirs <- list(
    list(dir = WT_DIR,  group = "WT"),
    list(dir = MUT_DIR, group = "MUT")
  )
  
  map_dfr(dirs, function(d) {
    files <- list.files(d$dir, pattern = "\\.dat$", full.names = TRUE)
    
    map_dfr(files, function(path) {
      fname <- basename(path)
      
      # Skip vec files
      if (str_detect(tolower(fname), "vec")) return(NULL)
      
      # Parse filename: Relay_OHelix_wt1.dat
      m <- str_match(fname, "^(.+)_(wt|mut)(\\d+)\\.dat$")
      if (is.na(m[1])) return(NULL)
      
      domain  <- m[2]
      sim_num <- m[4]
      
      if (!domain %in% DOMAINS) return(NULL)
      
      parse_angle_file(path, d$group, sim_num, domain)
    })
  }) |>
    mutate(group = factor(group, levels = c("WT", "MUT")))
}

# ── 3. Build per-simulation average df ---------------------------------------
build_sim_avg <- function(df) {
  df |>
    group_by(group, sim, domain) |>
    summarise(avg_angle = mean(angle), .groups = "drop") |>
    mutate(group = factor(group, levels = c("WT", "MUT")))
}

# ── 4. Build summary df (mean ± SEM) -----------------------------------------
build_summary <- function(df_sim) {
  df_sim |>
    group_by(group, domain) |>
    summarise(
      mean_angle = mean(avg_angle),
      sd_angle   = sd(avg_angle),
      se_angle   = sd(avg_angle) / sqrt(n()),
      .groups    = "drop"
    )
}

# ── 5. Plot -------------------------------------------------------------------
plot_helix_angles <- function(df_sim,
                              title = "Relay — O-Helix Angle") {
  df_avg <- build_summary(df_sim)
  y_max  <- max(df_avg$mean_angle + df_avg$se_angle, na.rm = TRUE) * 1.15
  y_min  <- min(df_avg$mean_angle - df_avg$se_angle, na.rm = TRUE) * 0.95
  
  p <- ggplot() +
    geom_col(data = df_avg,
             aes(x = domain, y = mean_angle, fill = group),
             position = position_dodge(0.7), width = 0.6,
             colour = "black", linewidth = 0.3) +
    geom_errorbar(data = df_avg,
                  aes(x     = domain,
                      ymin  = mean_angle - se_angle,
                      ymax  = mean_angle + se_angle,
                      group = group),
                  position = position_dodge(0.7), width = 0.2) +
    geom_point(data = df_sim,
               aes(x     = domain,
                   y     = avg_angle,
                   group = group,
                   color = group),
               position    = position_jitterdodge(jitter.width = 0.05,
                                                  dodge.width  = 0.7),
               size        = 2, shape = 16, show.legend = FALSE) +
    stat_compare_means(data        = df_sim,
                       aes(x      = domain,
                           y      = avg_angle,
                           group  = group),
                       method      = "t.test",
                       method.args = list(var.equal = TRUE),
                       label       = "p.signif",
                       hide.ns     = TRUE,
                       size        = 6,
                       fontface    = "bold",
                       inherit.aes = FALSE) +
    scale_fill_manual(values  = group_colors,       name = NULL) +
    scale_color_manual(values = group_colors_ticks, name = NULL) +
    labs(title = title,
         x     = NULL,
         y     = "Angle (degrees)") +
    coord_cartesian(ylim = c(y_min, y_max)) +
    theme_pubr(base_size = 14) +
    theme(text            = element_text(size = 14, face = "bold"),
          plot.title      = element_text(hjust = 0.5),
          axis.text.x     = element_text(angle = 0, hjust = 0.5),
          legend.position = "top")
  
  print(p)
  ggsave(file.path(SAVE_DIR, "helix_angles_relay_ohelix.png"),
         plot = p, width = 6, height = 6, dpi = 500)
}

# ── Run -----------------------------------------------------------------------
message("Loading data...")
df     <- collect_all_data()
df_sim <- build_sim_avg(df)

message("Loaded ", nrow(df), " rows | ",
        n_distinct(df$domain), " domains | ",
        n_distinct(df$sim),    " simulations")

print(df_sim)

plot_helix_angles(df_sim)

message("Done! Plot saved in '", SAVE_DIR, "'")