# Contact Domain Analysis ------------------------------------------------------
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

# Domain background shading colors --------------------------------------------
# Colors matched exactly to PyMOL structure coloring scheme.
# Edit hex codes to adjust. Color picker: https://htmlcolorcodes.com
# domain_colors <- c(
#   "P Loop"      = "#B8B9FF",
#   "Purine Loop" = "#B8B9FF",
#   "SH1"         = "#2E2E8F",
#   "S1"          = "#FF0000",
#   "S2"          = "#FF702E",
#   "Loop 2"      = "#AF3800",
#   "Loop 3"      = "#AF3800",
#   "HLH"         = "#AF3800",
#   "O-Helix"     = "#C8EAEE",
#   "Relay Helix" = "#C6E0F1",
#   "Upper 50kDa" = "#41b6c4",
#   "Lower 50kDa" = "#2c7fb8",
#   "ADP_Pi"      = "#41b6c4",
#   "Loop 4"      = "#AF3800",
#   "Ploop"       = "#B8B9FF",
#   "Purine"      = "#B8B9FF",
#   " "           = "#F0F0F0"
# )

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
  "Loop 4"      = "#E07050",
  "N-terminus"  = "#f2e9dd"
)

# User Settings ----------------------------------------------------------------
CSV_FILE <- "D:/Projects/MAB_project/Contact_Analysis/original_analysis/output/contact_diffs_all_d239n.csv"
SAVE_DIR <- "D:/Projects/MAB_project/Contact_Analysis/plots"

# Structural element residue ranges --------------------------------------------
ELEMENTS <- list(
  "S1"          = list(c(232, 244)),
  "SH1"         = list(c(674, 685)),
  "Ploop"       = list(c(179, 184)),
  "Purine"      = list(c(126, 131)),
  "ADP_Pi"      = list(c(852, 853)),
  "S2"          = list(c(462, 472)),
  "L2"          = list(c(624, 647)),
  "L3"          = list(c(567, 577)),
  "L4"          = list(c(361, 378)),
  "Upper 50kDa" = list(c(215, 231), c(266, 453), c(604, 621)),
  "Lower 50kDa" = list(c(645, 665), c(472, 590)),
  "N-terminus"  = list(c(113,123), c(666,671),c(170,175),c(454,460), c(244,265))
)

# Domains to plot
DOMAINS <- list(
  "S1"     = "S1",
  "SH1"    = "SH1",
  "Ploop"  = "Ploop",
  "Purine" = "Purine",
  "S2"     = "S2",
  "L2"     = "L2",
  "L3"     = "L3",
  "U50"    = "Upper 50kDa",
  "L50"    = "Lower 50kDa"
)

# Helper: which element does a residue belong to?
assign_element <- function(res) {
  for (elem_name in names(ELEMENTS)) {
    ranges <- ELEMENTS[[elem_name]]
    if (any(map_lgl(ranges, ~ res >= .x[1] & res <= .x[2]))) {
      return(elem_name)
    }
  }
  return(" ")
}

# Load CSV ---------------------------------------------------------------------
raw <- read_csv(CSV_FILE, show_col_types = FALSE)
names(raw)[1] <- "contact"
names(raw)[2:4]  <- c("wt1", "wt2", "wt3")
names(raw)[5:7]  <- c("m1",  "m2",  "m3")
names(raw)[10]   <- "pval"

raw <- raw %>%
  mutate(
    res1 = as.integer(str_extract(contact, "^[0-9]+")),
    res2 = as.integer(str_extract(contact, "[0-9]+$"))
  )

# Helper: check if residue is in element
in_element <- function(res, elem_name) {
  ranges <- ELEMENTS[[elem_name]]
  any(map_lgl(ranges, ~ res >= .x[1] & res <= .x[2]))
}

# Create output dir ------------------------------------------------------------
if (!is.null(SAVE_DIR)) dir.create(SAVE_DIR, recursive = TRUE, showWarnings = FALSE)

# Plotting function — one plot per domain --------------------------------------
plot_domain <- function(domain_name, primary_element) {

  df <- raw %>%
    filter(
      map_lgl(res1, in_element, primary_element) |
      map_lgl(res2, in_element, primary_element)
    ) %>%
    mutate(
      primary_res   = if_else(map_lgl(res1, in_element, primary_element), res1, res2),
      partner_res   = if_else(map_lgl(res1, in_element, primary_element), res2, res1),
      partner_elem  = map_chr(partner_res, assign_element),
      contact_label = paste0(primary_res, "-", partner_res)
    ) %>%
    filter(pval < 0.05)

  if (nrow(df) == 0) {
    message("Skipping '", domain_name, "' — no significant contacts found.")
    return(invisible(NULL))
  }

  # Reshape to long
  df_long <- df %>%
    pivot_longer(cols = c(wt1, wt2, wt3, m1, m2, m3),
                 names_to = "rep", values_to = "contact_pct") %>%
    mutate(genotype = factor(
      if_else(startsWith(rep, "w"), "WT", "D239N"),
      levels = c("WT", "D239N")
    ))

  # Average + SEM per contact
  df_avg <- df_long %>%
    group_by(contact, contact_label, primary_res, partner_res, partner_elem, pval, genotype) %>%
    summarise(
      mean_pct = mean(contact_pct, na.rm = TRUE),
      se_pct   = sd(contact_pct,   na.rm = TRUE) / sqrt(n()),
      .groups  = "drop"
    ) %>%
    mutate(
      pval_label = case_when(
        pval < 0.001 ~ "***",
        pval < 0.01  ~ "**",
        pval < 0.05  ~ "*",
        TRUE          ~ ""
      )
    )

  # Sort contacts by partner element then residue number
  contact_order <- df %>%
    arrange(primary_res, partner_res) %>%
    select(contact_label, partner_res) %>%
    distinct() %>%
    mutate(partner_elem = map_chr(partner_res, assign_element)) %>%
    arrange(partner_elem, partner_res) %>%
    pull(contact_label) %>%
    unique()

  x_map   <- setNames(seq_along(contact_order), contact_order)
  df_avg  <- df_avg  %>% mutate(x_pos = x_map[contact_label])
  df_long <- df_long %>% mutate(x_pos = x_map[contact_label])

  # Star positions — all on same y line
  y_max   <- max(df_avg$mean_pct + df_avg$se_pct, na.rm = TRUE)
  y_range <- y_max * 1.20
  y_stars <- y_max * 1.05

  pval_df <- df_avg %>%
    group_by(contact_label, x_pos, pval, pval_label) %>%
    summarise(.groups = "drop") %>%
    filter(pval_label != "") %>%
    mutate(y_pos = y_stars)

  # Background shading per partner element using domain_colors
  bg_df <- df_avg %>%
    select(contact_label, x_pos, partner_elem) %>%
    distinct() %>%
    group_by(partner_elem) %>%
    summarise(
      xmin  = min(x_pos) - 0.5,
      xmax  = max(x_pos) + 0.5,
      x_mid = (min(x_pos) + max(x_pos)) / 2,
      .groups = "drop"
    ) %>%
    mutate(bg_fill = scales::alpha(
      coalesce(domain_colors[partner_elem], "#F0F0F0"), 0.35
    ))

  # Separator lines between residue groups
  vline_pos <- seq(1.5, length(contact_order) - 0.5, by = 1)

  p <- ggplot() +
    annotate("rect",
             xmin  = bg_df$xmin, xmax = bg_df$xmax,
             ymin  = -Inf,        ymax = Inf,
             fill  = bg_df$bg_fill) +
    geom_text(data        = bg_df,
              aes(x = xmin + 0.1, y = y_range * 0.98, label = partner_elem),
              inherit.aes = FALSE,
              size = 6, fontface = "bold", hjust = 0, vjust = 1,
              color = "black") +
    geom_col(data     = df_avg,
             aes(x = x_pos, y = mean_pct, fill = genotype),
             position = position_dodge(width = 0.8), width = 0.8,
             colour = "black", linewidth = 0.3) +
    geom_errorbar(data = df_avg,
                  aes(x     = x_pos,
                      ymin  = mean_pct - se_pct,
                      ymax  = mean_pct + se_pct,
                      group = genotype),
                  position  = position_dodge(width = 0.8),
                  width     = 0.2, linewidth = 0.5) +
    geom_point(data        = df_long,
               aes(x = x_pos, y = contact_pct,
                   group = genotype, color = genotype),
               position    = position_jitterdodge(jitter.width = 0.05,
                                                  dodge.width  = 0.8),
               size        = 2, shape = 16,
               show.legend = FALSE) +
    geom_vline(xintercept = vline_pos,
               color = "darkgrey", linewidth = 0.4, linetype = "dashed") +
    geom_text(data        = pval_df,
              aes(x = x_pos, y = y_pos, label = pval_label),
              inherit.aes = FALSE,
              size = 8, vjust = 0) +
    scale_x_continuous(breaks = seq_along(contact_order),
                       labels = contact_order) +
    scale_y_continuous(expand = c(0, 0)) +
    coord_cartesian(ylim = c(0, y_range), clip = "off") +
    scale_fill_manual(values  = genotype_color_bar, name = NULL) +
    scale_color_manual(values = genotype_color_ticks, name = NULL) +
    labs(title = paste0("Contact Analysis of ", domain_name),
         x     = "Residue-Residue Pairs",
         y     = "Average Percentage of Frames (%)") +
    theme_pubr(base_size = 18) +
    theme(
      text                 = element_text(size = 18, face = "bold"),
      axis.title           = element_text(size = 18, face = "bold"),
      axis.text            = element_text(size = 18, face = "bold"),
      axis.text.x          = element_text(size = 18, face = "bold",
                                          angle = 0, hjust = 0.5),
      axis.text.y          = element_text(size = 18, face = "bold"),
      legend.text          = element_text(size = 18, face = "bold"),
      plot.title           = element_text(hjust = 0.5),
      legend.position      = c(1, 0.98),
      legend.justification = c(1, 1),
      legend.background    = element_rect(fill = "white", colour = "grey80"),
      plot.margin          = margin(t = 10, r = 10, b = 20, l = 10)
    ) +
    guides(fill = guide_legend(ncol = 1))

  print(p)

  if (!is.null(SAVE_DIR)) {
    filename <- file.path(SAVE_DIR, paste0("contact_", domain_name, ".png"))
    ggsave(filename, plot = p, width = 18, height = 10, dpi = 500)
    message("Saved: ", filename)
  }
}

# Run all domains --------------------------------------------------------------
for (domain_name in names(DOMAINS)) {
  message("Plotting domain: ", domain_name)
  plot_domain(domain_name, DOMAINS[[domain_name]])
}

message("Analysis complete! Plots saved in '", SAVE_DIR, "'")