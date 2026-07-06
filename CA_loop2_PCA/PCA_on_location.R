library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(purrr)
library(ggplot2)
library(ggpmisc)
library(ggpubr)

# #Create directory
#  dir.create("original_script/csv_files", recursive = TRUE, showWarnings = FALSE)
# 
#  read_in_pqr <- function(file_path, file_name) {
# 
#    df <- read_table(file_path, col_names = FALSE, show_col_types = FALSE)
# 
#    # Keep only ATOM/HETATM rows
#    df <- df[df$X1 %in% c("ATOM", "HETATM"), ]
# 
#    # Directly assign columns (no separate/unite)
#    df <- df %>%
#      transmute(
#        ATOM        = X1,
#        atom_number = as.numeric(X2),
#        atom_name   = X3,
#        res_name    = X4,
#        chain       = X5,
#        res_number  = as.numeric(X6),
#        x           = as.numeric(X7),
#        y           = as.numeric(X8),
#        z           = as.numeric(X9),
#        filename    = file_name
#      )
# 
#    return(df)
#  }
# 
#  file_names <- list.files("D:/Projects/MAB_project/CM_Loop2/delphi_pqr/",
#                           pattern = "\\.pqr$", full.names = TRUE)
# 
# df <- map_dfr(file_names, ~ read_in_pqr(.x, basename(.x)), .progress = TRUE)
# 
# write_csv(df, "original_script/csv_files/all_pdbs_df.csv")

df <- read_csv("original_script/csv_files/all_pdbs_df.csv")

#Filter to loop 2
df_loop2 <- df %>%
  filter(res_number >= 621,
         res_number <= 646)

write_csv(df_loop2, "original_script/csv_files/all_pdbs_df_loop2.csv")

df_loop2 <- read_csv("original_script/csv_files/all_pdbs_df_loop2.csv")

df_ca <- df_loop2 %>% 
  filter(atom_name == "CA") %>% 
  mutate(res_id = as.character(res_number))

df_ca_long <- df_ca %>% 
  pivot_longer(cols = c(x,y,z),
               names_to = 'xyz',
               values_to = 'cords') %>% 
  mutate(point_id = paste0(res_id, '_', xyz))

df_ca_wide <- df_ca_long %>% 
  select(filename, point_id, cords) %>% 
  pivot_wider(names_from = point_id, values_from = cords)

# Perform PCA
pca_loop2 <- prcomp(df_ca_wide %>% select(-filename),
                   scale. = F)

# Explore results
summary(pca_loop2)
#fviz_eig(pca_loop2)
#ggbiplot(pca_loop2)

variance_explained <- ((pca_loop2$sdev^2 / sum(pca_loop2$sdev^2))*100) %>% round(2)


# Extract the principal component scores
scores <- pca_loop2$x
scores <- scores %>% as_tibble()
weights <- pca_loop2$rotation
weights <- weights %>% as_tibble() %>% 
  mutate(pc_inputs = pca_loop2$rotation %>% rownames())

#put back together with metadata 
df_pca <- bind_cols(df_ca_wide %>% select(filename), scores)

#Pull out data from file name
df_pca <- df_pca %>%
  mutate(
    Genotype = toupper(str_extract(filename, "^[^_]+")),
    sim_number = as.numeric(str_extract(filename, "(?<=_sim)\\d+")),
    Simulation = paste0(Genotype, "_", sim_number),
    time = as.numeric(str_extract(filename, "(?<=frame)\\d{3}"))
  )

df_pca <- df_pca %>% 
  mutate(Genotype = toupper(Genotype),
         Simulation = paste0(Genotype, '_', sim_number))

df_binding_force <- read.csv("D:/Projects/MAB_project/Delphi_Force/csv/results_formatted.csv")

df_dist <- read.csv("D:/Projects/MAB_project/CM_Loop2/csv/loop2lys_to_actNterm.csv")  
  
df_pca %>% 
  select(Simulation)

df_force_distance <- df_binding_force %>%
  left_join(df_dist, by = c("sim", "Genotype", "time"))

#Matching column names between pca and force_distance
df_pca_2 <- df_pca %>%
  mutate(file_base = str_remove(basename(filename), "\\.[^.]+$"))  # remove .pqr

df_force_distance_2 <- df_force_distance %>%   # or df_force_distance / your distance df
  mutate(file_base = str_remove(basename(file_name), "\\.[^.]+$")) # remove .residue

df_pca_force_dist <- df_force_distance_2 %>%
  left_join(df_pca_2, by = "file_base")

#Treat genotypes and simulation as factor
df_pca_force_dist <- df_pca_force_dist %>% 
  mutate(Simulation = factor(Simulation, levels = c("WT_1", "WT_2", "WT_3",
                                                    "D239N_1", "D239N_2", "D239N_3",
                                                    "K637E_1", "K637E_2", "K637E_3")),
         Genotype = factor(Genotype.x, levels = c("WT", "D239N", "K637E")))


# Save the PCA data and metadata ------------------------------------------
write_csv(df_pca_force_dist, "D:/Projects/MAB_project/CA_Loop2_PCA_Analysis/original_script/csv_files/df_pca_force_dist.csv")
write_csv(variance_explained %>% tibble() %>% rename(variance_explained = "."),
          "D:/Projects/MAB_project/CA_Loop2_PCA_Analysis/original_script/csv_files/variance_explained.csv")

#Look into the waving of Loop 2
df_range <- df_pca_force_dist %>% 
  select(Simulation, PC1, PC2, PC3, PC4, PC5, time.x, dist, binding_force, Genotype, filename) %>% 
  group_by(Simulation, Genotype) %>% 
  reframe(PC2max = max(PC2),
          PC2min = min(PC2),
          distmax = max(dist),
          distmin = min(dist),
          binding_forcemax = max(binding_force),
          binding_forcemin = min(binding_force)) %>% 
  mutate(PC2range = PC2max - PC2min,
         distrange = distmax - distmin,
         binding_forcerange = binding_forcemax - binding_forcemin) %>% 
  arrange(PC2range) #PC2 seems to account for the "dist" measurement

#Find the max and min of the different PCs for the loop 2 paper figure 
df_pca_force_dist %>% 
  select(Simulation, PC1, PC2, PC3, PC4, PC5, time.x, dist, binding_force, Genotype, filename) %>% 
  filter(Simulation == "WT_1") %>% 
  arrange(PC3) %>% head

#Graph PC1 vs. loop 2 to actin n-term distance
ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = dist, y = PC1, color = Simulation),
             size = 3)+
  
  stat_poly_line(data = df_pca_force_dist,
                 aes(x = dist, y =  PC1),
                 formula = y~x,
                 color = "red",
                 linewidth = 2)+
  stat_poly_eq(data = df_pca_force_dist,
               aes(x = dist, y =  PC1),
               formula = y ~ x,
               size = 10,
               label.x = "right")+
  labs(title = "Dist Act N-term vs. PC1", x = "Distance (Å)", y = "PC1") +
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_color_manual(values = c('lightgrey', 'darkgrey', 'black', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))

ggsave("original_script/plots_2D/PC1vsDist.png",
       width = 8, height = 8, dpi = 500)

#Graph PC2 vs. loop 2 to actin n-term distance
ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = dist, y = PC2, color = Simulation),
             size = 3)+
  
  stat_poly_line(data = df_pca_force_dist,
                 aes(x = dist, y =  PC2),
                 formula = y~x,
                 color = "red",
                 linewidth = 2)+
  stat_poly_eq(data = df_pca_force_dist,
               aes(x = dist, y =  PC2),
               formula = y ~ x,
               size = 10,
               label.x = "right")+
  labs(title = "Dist Act N-term vs. PC2", x = "Distance (Å)", y = "PC2") +
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_color_manual(values = c('lightgrey', 'darkgrey', 'black', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))

ggsave("original_script/plots_2D/PC2vsDist.png",
       width = 8, height = 8, dpi = 500)

# Save out data arranged by PC1 -------------------------------------------
WT_1_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_1") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_1_PC1_arrange, "original_script/csv_files/WT_1_PC1_arrange.csv")

WT_2_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_2") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_2_PC1_arrange, "original_script/csv_files/WT_2_PC1_arrange.csv")

WT_3_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_3") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_3_PC1_arrange, "original_script/csv_files/WT_3_PC1_arrange.csv")

D239N_1_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_1") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_1_PC1_arrange, "original_script/csv_files/D239N_1_PC1_arrange.csv")

D239N_2_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_2") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_2_PC1_arrange, "original_script/csv_files/D239N_2_PC1_arrange.csv")

D239N_3_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_3") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_3_PC1_arrange, "original_script/csv_files/D239N_3_PC1_arrange.csv")

K637E_1_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_1") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_1_PC1_arrange, "original_script/csv_files/K637E_1_PC1_arrange.csv")

K637E_2_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_2") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_2_PC1_arrange, "original_script/csv_files/K637E_2_PC1_arrange.csv")

K637E_3_PC1_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_3") %>% 
  arrange(PC1) %>% 
  select(time.x,PC1,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_3_PC1_arrange, "original_script/csv_files/K637E_3_PC1_arrange.csv")

# Save out data arranged by PC2 -------------------------------------------
WT_1_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_1") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_1_pc2_arrange, "original_script/csv_files/WT_1_pc2_arrange.csv")

WT_2_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_2") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_2_pc2_arrange, "original_script/csv_files/WT_2_pc2_arrange.csv")

WT_3_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_3") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_3_pc2_arrange, "original_script/csv_files/WT_3_pc2_arrange.csv")

D239N_1_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_1") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_1_pc2_arrange, "original_script/csv_files/D239N_1_pc2_arrange.csv")

D239N_2_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_2") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_2_pc2_arrange, "original_script/csv_files/D239N_2_pc2_arrange.csv")

D239N_3_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_3") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_3_pc2_arrange, "original_script/csv_files/D239N_3_pc2_arrange.csv")

K637E_1_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_1") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_1_pc2_arrange, "original_script/csv_files/K637E_1_pc2_arrange.csv")

K637E_2_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_2") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_2_pc2_arrange, "original_script/csv_files/K637E_2_pc2_arrange.csv")

K637E_3_pc2_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_3") %>% 
  arrange(PC2) %>% 
  select(time.x,PC2,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_3_pc2_arrange, "original_script/csv_files/K637E_3_pc2_arrange.csv")


# Save out data arranged by PC3 -------------------------------------------
WT_1_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_1") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_1_PC3_arrange, "original_script/csv_files/WT_1_PC3_arrange.csv")

WT_2_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_2") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_2_PC3_arrange, "original_script/csv_files/WT_2_PC3_arrange.csv")

WT_3_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "WT_3") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(WT_3_PC3_arrange, "original_script/csv_files/WT_3_PC3_arrange.csv")

D239N_1_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_1") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_1_PC3_arrange, "original_script/csv_files/D239N_1_PC3_arrange.csv")

D239N_2_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_2") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_2_PC3_arrange, "original_script/csv_files/D239N_2_PC3_arrange.csv")

D239N_3_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "D239N_3") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(D239N_3_PC3_arrange, "original_script/csv_files/D239N_3_PC3_arrange.csv")

K637E_1_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_1") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_1_PC3_arrange, "original_script/csv_files/K637E_1_PC3_arrange.csv")

K637E_2_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_2") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_2_PC3_arrange, "original_script/csv_files/K637E_2_PC3_arrange.csv")

K637E_3_PC3_arrange <- df_pca_force_dist %>% 
  filter(Simulation == "K637E_3") %>% 
  arrange(PC3) %>% 
  select(time.x,PC3,dist, filename) %>% 
  mutate(index = row_number())

write_csv(K637E_3_PC3_arrange, "original_script/csv_files/K637E_3_PC3_arrange.csv")



# Find middle value for each PC -------------------------------------------
rangePC = .05

df_pca_force_dist %>% 
  select(filename, PC1, PC2, PC3) %>% 
          filter(PC1 > median(PC1)-rangePC & PC1 < median(PC1)+rangePC)

df_pca_force_dist %>% 
  select(filename, PC1, PC2, PC3) %>% 
  filter(PC2 > median(PC2)-rangePC & PC2 < median(PC2)+rangePC)
          
df_pca_force_dist$PC1 %>% median

# plots looking at PC2 more closely  --------------------------------------


#PC2 by Genotype 
ggplot()+
  geom_point(data = df_pca_force_dist, 
             aes(x = Genotype, y = PC2, color = Simulation), 
             size = 1,
             shape = 1,
             alpha = .3,
             position = position_jitterdodge(jitter.width = .4,
                                             dodge.width = .6,))+ 

  stat_summary(data = df_pca_force_dist,
               fun = mean,
               geom = "point",
               aes(x = Genotype, y = PC2, color = Simulation),
               size = 7, alpha = 1, position = position_dodge(.6))+
  
  labs(title = "PC2 by Genotype", x = "", y = "PC2") +
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_manual(values = c('lightgrey', 'darkgrey', 'black', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))
ggsave("original_script/plots_2D/PC2byGenotype.png",
         width = 5, height = 5, dpi = 500)


#Binding force range by Genotype 
ggplot()+
  # geom_point(data = df_range, 
  #            aes(x = Genotype, y = binding_forcerange, color = Simulation), 
  #            size = 1,
  #            shape = 1,
  #            alpha = .3,
  #            position = position_jitterdodge(jitter.width = .4,
  #                                            dodge.width = .6,))+ 
  
  stat_summary(data = df_range,
               fun = mean,
               geom = "bar",
               aes(x = Genotype, y = binding_forcerange, fill = Simulation),
               alpha = 1, color = "black", position = position_dodge(.9))+
  
  labs(title = "Binding Force Range", x = "", y = "Force (nN)") +
  theme_pubr() +
  guides(fill=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_manual(values = c('lightgrey', 'darkgrey', 'black', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))+
  scale_fill_manual(values = c('lightgrey', 'darkgrey', 'black', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))

ggsave("original_script/plots/FroceRangebyGenotype.png",
       width = 8, height = 8, dpi = 500)

df
# PC1 vs. PC2 -------------------------------------------------------------
#Graph PC1 vs. PC2
ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC1, y = PC2, color = Simulation),
             size = 2
             )+
  
  labs(title = NULL, 
       x = paste0("PC1 [", variance_explained[1], "% variance]"), 
       y = paste0("PC2 [", variance_explained[2], "% variance]")
       )+
  theme_pubr() +
  # guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25),
        legend.position = "none") +
  scale_color_manual(values = c('lightgrey', 'darkgrey', 'black', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))

ggsave("original_script/plots_2D/PC1_PC2_sim.png",
       width = 5, height = 5, dpi = 500)


ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC1, y = PC2, color = dist),
             size = 2
  )+
  
  labs(title = NULL, 
       x = paste0("PC1 [", variance_explained[1], "% variance]"), 
       y = paste0("PC2 [", variance_explained[2], "% variance]")
  )+
  theme_pubr() +
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25))+
  scale_color_gradient(low = "darkblue", high = "yellow", na.value = NA, name = "Loop 2 Dist (Å)")

ggsave("original_script/plots_2D/PC1_PC2_dist.png",
       width = 5, height = 5, dpi = 500)

ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC1, y = PC2, color = binding_force),
             size = 2
  )+
  
  labs(title = NULL, 
       x = paste0("PC1 [", variance_explained[1], "% variance]"), 
       y = paste0("PC2 [", variance_explained[2], "% variance]")
  )+
  theme_pubr() +
  scale_color_gradient(low = "darkblue", high = "yellow", na.value = NA, name = "Force (nN)")+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25),
        legend.text = element_text(size = 15))
  

ggsave("original_script/plots_2D/PC1_PC2_binding_force.png",
       width = 5, height = 5, dpi = 500)


# PC1 vs. PC3 -------------------------------------------------------------
#Graph PC1 vs. PC3
ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC1, y = PC3, color = Simulation),
             size = 2
  )+
  
  labs(title = NULL, 
       x = paste0("PC1 [", variance_explained[1], "% variance]"), 
       y = paste0("PC3 [", variance_explained[3], "% variance]")
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25),
        legend.position = "none") +
  scale_color_manual(values = c('lightgrey', 'darkgrey', 'black', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))

ggsave("original_script/plots_2D/PC1_PC3_sim.png",
       width = 5, height = 5, dpi = 500)


ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC1, y = PC3, color = dist),
             size = 2
  )+
  
  labs(title = NULL, 
       x = paste0("PC1 [", variance_explained[1], "% variance]"), 
       y = paste0("PC3 [", variance_explained[3], "% variance]")
  )+
  theme_pubr() +
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25))+
  scale_color_gradient(low = "darkblue", high = "yellow", na.value = NA, name = "Loop 2 Dist (Å)")

ggsave("original_script/plots_2D/PC1_PC3_dist.png",
       width = 5, height = 5, dpi = 500)

ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC1, y = PC3, color = binding_force),
             size = 2
  )+
  
  labs(title = NULL, 
       x = paste0("PC1 [", variance_explained[1], "% variance]"), 
       y = paste0("PC3 [", variance_explained[3], "% variance]")
  )+
  theme_pubr() +
  scale_color_gradient(low = "darkblue", high = "yellow", na.value = NA, name = "Force (nN)")+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25),
        legend.text = element_text(size = 15))

ggsave("original_script/plots_2D/PC1_PC3_binding_force.png",
       width = 5, height = 5, dpi = 500)

# PC2 vs. PC3 -------------------------------------------------------------
#Graph PC2 vs. PC3
ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC2, y = PC3, color = Simulation),
             size = 2
  )+
  
  labs(title = NULL, 
       x = paste0("PC2 [", variance_explained[2], "% variance]"), 
       y = paste0("PC3 [", variance_explained[3], "% variance]")
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25),
        legend.position = "none") +
  scale_color_manual(values = c('lightgrey', 'darkgrey', 'black', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))

ggsave("original_script/plots_2D/PC2_PC3_sim.png",
       width = 5, height = 5, dpi = 500)


ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC2, y = PC3, color = dist),
             size = 2
  )+
  
  labs(title = NULL, 
       x = paste0("PC2 [", variance_explained[2], "% variance]"), 
       y = paste0("PC3 [", variance_explained[3], "% variance]")
  )+
  theme_pubr() +
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25))+
  scale_color_gradient(low = "darkblue", high = "yellow", na.value = NA, name = "Loop 2 Dist (Å)")

ggsave("original_script/plots_2D/PC2_PC3_dist.png",
       width = 5, height = 5, dpi = 500)

ggplot()+
  geom_point(data = df_pca_force_dist,
             aes(x = PC2, y = PC3, color = binding_force),
             size = 2
  )+
  
  labs(title = NULL, 
       x = paste0("PC2 [", variance_explained[2], "% variance]"), 
       y = paste0("PC3 [", variance_explained[3], "% variance]")
  )+
  theme_pubr() +
  scale_color_gradient(low = "darkblue", high = "yellow", na.value = NA, name = "Force (nN)")+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=25),
        legend.text = element_text(size = 15))

ggsave("original_script/plots_2D/PC2_PC3_binding_force.png",
       width = 5, height = 5, dpi = 500)


# Graph PC Weights  -------------------------------------------------------
# weights <- 
weights <- weights %>% 
  mutate(xyz = substr(pc_inputs, nchar(pc_inputs), nchar(pc_inputs))) %>% 
  mutate(pc_inputs = substr(pc_inputs, 1, nchar(pc_inputs)-2)) %>% 
  mutate(aa = substr(pc_inputs, nchar(pc_inputs)-2, nchar(pc_inputs))) %>% 
  mutate(Charge = ifelse(aa == "LYS" | aa == "ARG" | aa == "HIS", "Pos",
                         ifelse(aa == "ASP" | aa == "GLU", "Neg",
                                "Neut"))) %>% 
  mutate(Charge = factor(Charge, levels = c("Neut", "Pos", "Neg")))

weights <- weights %>% 
  mutate(res_num = parse_number(pc_inputs))

#PC1
ggplot()+
  geom_col(data = weights,
           aes(x = res_num, y = PC1, fill = Charge))+
  
  labs(title = paste0("PC1 [", variance_explained[1], "% variance]"), 
       x = "Loop 2 AA", 
       y = "PC Vector Weight"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=35, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=35)) +
  scale_fill_manual(values = c('gray', 'green', 'blue'))+
  facet_wrap(~xyz)
  #ylim(-0.36,0.36)

ggsave("original_script/plots_weights/PC1.png",
       width = 20, height = 7, dpi = 500)


#PC2
ggplot()+
  geom_col(data = weights,
           aes(x = res_num, y = PC2, fill = Charge))+
  
  labs(title = paste0("PC2 [", variance_explained[2], "% variance]"), 
       x = "Loop 2 AA", 
       y = "PC Vector Weight"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=35, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=35)) +
  scale_fill_manual(values = c('gray', 'green', 'blue'))+
  facet_wrap(~xyz)
  #ylim(-0.36,0.36)

ggsave("original_script/plots_weights/PC2.png",
       width = 20, height = 7, dpi = 500)


#PC3
ggplot()+
  geom_col(data = weights,
           aes(x = res_num, y = PC3, fill = Charge))+
  
  labs(title = paste0("PC3 [", variance_explained[3], "% variance]"), 
       x = "Loop 2 AA", 
       y = "PC Vector Weight"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=35, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=35)) +
  scale_fill_manual(values = c('gray', 'green', 'blue'))+
  facet_wrap(~xyz)
  ##ylim(-0.36,0.36)

ggsave("original_script/plots_weights/PC3.png",
       width = 20, height = 7, dpi = 500)

#PC4
ggplot()+
  geom_col(data = weights,
           aes(x = res_num, y = PC4, fill = Charge))+
  
  labs(title = paste0("PC4 [", variance_explained[4], "% variance]"), 
       x = "Loop 2 AA", 
       y = "PC Vector Weight"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=35, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=35)) +
  scale_fill_manual(values = c('gray', 'green', 'blue'))+
  facet_wrap(~xyz)
  #ylim(-0.36,0.36)

ggsave("original_script/plots_weights/PC4.png",
       width = 20, height = 7, dpi = 500)

#PC5
ggplot()+
  geom_col(data = weights,
           aes(x = res_num, y = PC5, fill = Charge))+
  
  labs(title = paste0("PC5 [", variance_explained[5], "% variance]"), 
       x = "Loop 2 AA", 
       y = "PC Vector Weight"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=35, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=35)) +
  scale_fill_manual(values = c('darkgray', 'green', 'blue'))+
  facet_wrap(~xyz)
  #ylim(-0.36,0.36)

ggsave("original_script/plots_weights/PC5.png",
       width = 20, height = 7, dpi = 500)


#PC6
ggplot()+
  geom_col(data = weights,
           aes(x = res_num, y = PC6, fill = Charge))+
  
  labs(title = paste0("PC6 [", variance_explained[6], "% variance]"), 
       x = "Loop 2 AA", 
       y = "PC Vector Weight"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=35, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(size=35)) +
  scale_fill_manual(values = c('darkgray', 'blue', 'red'))+
  facet_wrap(~xyz)
  #ylim(-0.36,0.36)

ggsave("original_script/plots_weights/PC6.png",
       width = 20, height = 7, dpi = 500)
   




# Graph PC1-2 histograms --------------------------------------------------


#Density plot PC1
ggplot()+
  geom_density(data = df_pca_force_dist,
                 aes(PC1, fill = Genotype),
               alpha = .7)+
  
  labs(title = NULL, 
       x = paste0("PC1 [", variance_explained[1], "% variance]"), 
       y = "Density"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  scale_fill_manual(values = c('darkgrey', 'green', 'blue'))

ggsave("original_script/plots_density/PC1_density.png",
       width = 5, height = 4, dpi = 500)

#Density plot PC2
ggplot()+
  geom_density(data = df_pca_force_dist,
               aes(PC2, fill = Genotype),
               alpha = .7)+
  
  labs(title = NULL, 
       x = paste0("PC2 [", variance_explained[2], "% variance]"), 
       y = "Density"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  scale_fill_manual(values = c('darkgrey', 'green', 'blue'))

ggsave("original_script/plots_density/PC2_density.png",
       width = 5, height = 4, dpi = 500)

#Density plot PC3
ggplot()+
  geom_density(data = df_pca_force_dist,
               aes(PC3, fill = Genotype),
               alpha = .7)+
  
  labs(title = NULL, 
       x = paste0("PC3 [", variance_explained[3], "% variance]"), 
       y = "Density"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=25, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  scale_fill_manual(values = c('darkgrey', 'green', 'blue'))

ggsave("original_script/plots_density/PC3_density.png",
       width = 5, height = 4, dpi = 500)

# Scree Plot  -------------------------------------------------------------

df_variance <- tibble(variance = variance_explained,
       PC = paste0("PC", 1:78)) %>% 
  mutate(PC = factor(PC, levels = paste0("PC", 1:78)))

ggplot()+
  geom_col(data = df_variance,
             aes(x = PC, y = variance))+
  
  labs(title = NULL, 
       x = NULL, 
       y = "Variance Explained (%)"
  )+
  theme_pubr() +
  guides(color=guide_legend(nrow=3))+
  theme(text = element_text(size=20, face = "bold"),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(size = 10, angle = 45, hjust = 1),
        legend.position = "none") +
  xlim(paste0("PC", 1:20))

ggsave("original_script/plots/scree_plot.png",
       width = 10, height = 4, dpi = 500)

# Create a Q reduced and y file for PCA 3D plot
dir.create("original_script/pca_output", showWarnings = FALSE)

Q_reduced <- df_pca_force_dist %>%
  dplyr::select(PC1, PC2, PC3)

write.csv(Q_reduced,
          "original_script/pca_output/Q_reduced_trimmed.csv",
          row.names = FALSE)

sim_levels <- c("WT_1","WT_2","WT_3",
                "D239N_1","D239N_2","D239N_3",
                "K637E_1","K637E_2","K637E_3")

y <- df_pca_force_dist %>%
  mutate(sim_id = as.integer(factor(Simulation, levels = sim_levels)) - 1) %>%
  dplyr::select(sim_id)

write.csv(y,
          "original_script/pca_output/y_trimmed.csv",
          row.names = FALSE)

dist_vec <- df_pca_force_dist %>%
  dplyr::select(dist)

write.csv(dist_vec,
          "original_script/pca_output/dist_trimmed.csv",
          row.names = FALSE)
