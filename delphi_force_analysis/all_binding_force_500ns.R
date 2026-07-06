#delphi_force_analysis in R to graph binding energy 

library(tidyverse)
library(ggpubr)
library(data.table)
library(dplyr)

# read in data ------------------------------------------------------------

# files <- list.files(path = "files_pqr", pattern = ".residue$", full.names = TRUE)
# 
# 
# read_last_value <- function(f) {
#   last_line <- read_csv(f, col_names = F) %>%
#     tail(1) %>%
#     as.character()
# 
#   as.numeric(str_extract(paste(last_line, collapse = " "), "[-+]?\\d+(?:\\.\\d+)?"))
# }
# 
# results <- data.frame(
#   file_name = basename(files),
#   binding_energy = vapply(files, read_last_value, numeric(1))
# )
# results <- results[!is.na(results$binding_energy), ]
# results <- as_tibble(results)
# 
# 
# write_csv(results, "files_csv/results.csv")
results <- read_csv("D:/Projects/MAB_project/Delphi_Force/Kalen_Full_csv/results.csv")

results %>% arrange(binding_energy) %>% head

# organize data -----------------------------------------------------------

#Pull out data from file name
df <- results %>% 
  mutate(Genotype = str_extract(file_name, "^[^_]+") %>% toupper(),
         Genotype = factor(Genotype, levels = c("WT", "D239N", "K637E"))) %>% 
  mutate(sim = str_extract(file_name, "(?<=sim)\\d+(?=_)") %>% as.numeric()) %>% 
  mutate(frame = str_extract(file_name, "(?<=frame)\\d+(?=_)") %>% as.numeric()) %>% 
  rename(time = frame) %>% 
  mutate(Sim_ID = paste0(Genotype, "_", sim),
         Sim_ID = factor(Sim_ID, levels = c("WT_1", "WT_2", "WT_3",
                                            "D239N_1", "D239N_2", "D239N_3",
                                            "K637E_1", "K637E_2", "K637E_3")))

# Convert binding energy to binding force 
#Convert 1kT/A = 41.14 nanoNewton and Convert to positive numbers for easier to read data
df <- df %>% 
  mutate(binding_force = binding_energy*-41.14) 


#Save formated data to be graphed with distance
write_csv(df, "D:/Projects/MAB_project/Delphi_Force/csv/results_formatted.csv")


df %>% 
  group_by(Genotype, sim) %>% 
  reframe(binding_force = mean(binding_force),
          n = n())

df_avg_gt <- df %>% 
  group_by(Genotype, time) %>% 
  reframe(binding_force = mean(binding_force))

df_avg <- df %>%
  group_by(Genotype) %>% 
  reframe(binding_force = mean(binding_force)) %>%
  mutate(mean_label = paste0("Mean = ", round(binding_force, 2), " nN"))
  
df_avg_simid <- df %>%
  group_by(Sim_ID) %>%
  summarise(binding_force = mean(binding_force), .groups = "drop") %>%
  mutate(mean_label = paste0("Mean = ", round(binding_force, 2), " nN"))

ggplot()+
  geom_line(data = df,
            aes(x=time, y=binding_force, color = Sim_ID, group = sim),
            linewidth = 1)+
  geom_hline(data = df_avg, 
             aes(yintercept = binding_force), 
             color = "black",
             linetype = "dashed")+
  geom_text(
    data = df_avg,
    aes(x = Inf, y = Inf, label = mean_label),
    hjust = 1.1, vjust = 1.5,
    inherit.aes = FALSE,
    size = 5, fontface = "bold")+

  
  scale_color_manual(values = c("black", "darkgray", "lightgray", 
                                "darkgreen", "green", "lightgreen",
                                "darkblue", "blue","lightblue"))+
  
  theme_pubr()+
  labs(title = NULL, x = "Time (ns)", y = "Binding Force (nN)") +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=18, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 0, hjust = .5)) +
  facet_wrap(~Genotype)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force/force_by_sim.png",
       width = 19, height = 6, dpi = 500)


ggplot()+
  geom_line(data = df_avg_gt,
            aes(x=time, y=binding_force, color = Genotype),
            linewidth = 1)+
  geom_hline(data = df_avg, 
             aes(yintercept = binding_force), 
             color = "black",
             linetype = "dashed")+
  geom_text(
    data = df_avg,
    aes(x = Inf, y = Inf, label = mean_label),
    hjust = 1.1, vjust = 1.5,
    inherit.aes = FALSE,
    size = 5, fontface = "bold")+
  
  scale_color_manual(values = c("black", "green", "blue"))+
  theme_pubr()+
  labs(title = NULL, x = "Time (ns)", y = "Binding Force (nN)") +
  theme(text = element_text(size=18, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 0, hjust = .5),
        legend.position = "none")+
  facet_wrap(~Genotype)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force/force_by_gt.png",
       width = 19, height = 6, dpi = 500)


ggplot()+
  geom_density(data = df_avg_gt,
               aes(x = binding_force, 
                   color = Genotype,
                   fill = Genotype), alpha = .5)+
  scale_color_manual(values = c("black", "green", "blue"))+
  scale_fill_manual(values = c("black", "green", "blue"))+
  
  labs(title = NULL, x = "Binding Force (nN)", y = "Density") +
  theme_pubr()+
  theme(text = element_text(size=18, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 0, hjust = .5),
        legend.position = "top")

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force/force_by_gt_density.png",
       width = 12, height = 6, dpi = 500)

ggplot()+
  geom_density(data = df,
               aes(x = binding_force, 
                   color = Sim_ID,
                   fill = Sim_ID,
                   alpha = .5))+
  scale_color_manual(values = c("black", "darkgray", "lightgray", 
                                "darkgreen", "green", "lightgreen",
                                "darkblue", "blue", "lightblue"))+
  scale_fill_manual(values = c("black", "darkgray", "lightgray", 
                                "darkgreen", "green", "lightgreen",
                                "darkblue", "blue", "lightblue"))+
  
  labs(title = NULL, x = "Binding Force (nN)", y = "Density") +
  theme_pubr()+
  theme(text = element_text(size=18, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 0, hjust = .5),
        legend.position = "none")+
  facet_wrap(~Genotype)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force/force_by_sim_density.png",
       width = 19, height = 6, dpi = 500)

ggplot() +
  geom_line(data = df,
            aes(x = time, y = binding_force, color = Sim_ID, group = Sim_ID),
            linewidth = 1) +
  geom_hline(data = df_avg_simid,
             aes(yintercept = binding_force),
             color = "black", linetype = "dashed") +
  geom_text(
    data = df_avg_simid,  # ✅ use Sim_ID means
    aes(x = Inf, y = Inf, label = mean_label),
    hjust = 1.1, vjust = 2.2,  # ✅ push down to avoid strip overlap
    inherit.aes = FALSE,
    size = 4, fontface = "bold", color = "black"
  ) +
  scale_color_manual(values = c(
    "WT_1"="black", "WT_2"="darkgray", "WT_3"="lightgray",
    "D239N_1"="darkgreen", "D239N_2"="green", "D239N_3"="lightgreen",
    "K637E_1"="darkblue", "K637E_2"="blue", "K637E_3"="lightblue"
  )) +
  theme_pubr() +
  labs(x = "Time (ns)", y = "Binding Force (nN)") +
  facet_wrap(~Sim_ID, ncol = 3) +
  coord_cartesian(clip = "off") +
  theme(text = element_text(size = 14, face = "bold"),
        legend.position = "none")

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force/split_by_simid.png",
       width = 19, height = 10, dpi = 500)


# Find average and representative frame  ---------------------------------
df_avg_sim <- df %>% 
  group_by(Genotype, sim) %>% 
  reframe(binding_force = mean(binding_force)) %>% 
  rename(binding_force_mean = binding_force)


full_join(df, df_avg_sim, by = c("Genotype", "sim")) %>% 
  mutate(diff_force = abs(binding_force - binding_force_mean)) %>% 
  group_by(Genotype, sim) %>% 
  slice_min(diff_force, n = 1) %>% 
  select(Genotype, sim, file_name, diff_force, binding_force, binding_force_mean)

df_avg_gt <- df %>% 
  group_by(Genotype) %>% 
  reframe(binding_force = mean(binding_force)) %>% 
  rename(binding_force_mean = binding_force)


full_join(df, df_avg_gt, by = c("Genotype")) %>% 
  mutate(diff_force = abs(binding_force - binding_force_mean)) %>% 
  group_by(Genotype) %>% 
  slice_min(diff_force, n = 1) %>% 
  select(Genotype, sim, file_name, diff_force, binding_force, binding_force_mean)

# Find the min and max for each Sim and their Frame
df_minmax_sim <- df %>%
  group_by(Genotype, sim, Sim_ID) %>%
  summarise(
    min_force = min(binding_force, na.rm = TRUE),
    min_frame  = time[which.min(binding_force)],
    max_force = max(binding_force, na.rm = TRUE),
    max_frame  = time[which.max(binding_force)],
    .groups = "drop"
  )
