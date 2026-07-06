#delphi_force_analysis in R to graph binding energy 
library(tidyverse)
library(ggpubr)
library(plotly)
library(readr)
library(processx)
library(data.table)
library(ggbreak)

wt_color = "black"
d239n_color = "green"
k637e_color = "blue"


# Read in data ---------------------------------------------------

# read_in_out_residue_fast <- function(path, file_name) {
#   # Read raw lines, skip header (line 1) and drop last 2 lines
#   lines <- readLines(path, warn = FALSE)
#   lines <- lines[-1]                     # drop header row
#   lines <- head(lines, -2)              # drop last 2 lines
#   lines <- lines[nzchar(trimws(lines))] # drop any blank lines
#   
#   # Fixed-width extraction using substr
#   data.table(
#     Residue    = substr(lines, 1,  3),
#     Chain      = substr(lines, 5,  5),
#     ID         = as.integer(substr(lines, 7,  10)),
#     Net_Charge = as.numeric(substr(lines, 19, 25)),
#     G          = as.numeric(substr(lines, 29, 35)),
#     Fx         = as.numeric(substr(lines, 39, 45)),
#     Fy         = as.numeric(substr(lines, 49, 55)),
#     Fz         = as.numeric(substr(lines, 59, 65)),
#     file_name  = file_name
#   )[, Residue_ID := paste0(ID, "_", Residue)]
# }
# 
# # Read all files
# filelist <- list.files("files_pqr/", pattern = "\\.residue$", full.names = FALSE)
# pathlist <- paste0("files_pqr/", filelist)
# 
# results <- lapply(seq_along(pathlist), function(i)
#   read_in_out_residue_fast(pathlist[i], filelist[i])
# )
# 
# df <- rbindlist(results) |> as_tibble()
# 
# #Pull out data from file name
# df <- df %>% 
#   mutate(Genotype = str_extract(file_name, "^[^_]+") %>% toupper(),
#          Genotype = factor(Genotype, levels = c("WT", "D239N", "K637E"))) %>% 
#   mutate(sim = str_extract(file_name, "(?<=sim)\\d+(?=_)") %>% as.numeric()) %>% 
#   mutate(frame = str_extract(file_name, "(?<=frame)\\d+(?=_)") %>% as.numeric()) %>% 
#   rename(time = frame) %>% 
#   mutate(Sim_ID = paste0(Genotype, "_", sim),
#          Sim_ID = factor(Sim_ID, levels = c("WT_1", "WT_2", "WT_3",
#                                             "D239N_1", "D239N_2", "D239N_3",
#                                             "K637E_1", "K637E_2", "K637E_3")))
# 
# #save data in CSV
# write_csv(df, "files_csv/results_residue.csv")
df <- read_csv("D:/Projects/MAB_project/Delphi_Force/Kalen_Full_csv/results_residue_formatted.csv")

df <- df %>% 
  mutate(Genotype = factor(Genotype, levels = c("WT", "D239N", "K637E")))

# Find important residues  ------------------------------------------------


# Convert binding energy to binding force 
df <- df %>% 
  mutate(binding_force = G*-41.14)  #Convert 1kT/A = 41.14 nanoNewton (nN) and make positive

df <- df %>% 
  mutate(Simulation = paste0(Genotype, "_", sim)) %>% 
  mutate(Simulation = factor(Simulation, 
                             levels = c("WT_1", "WT_2", "WT_3", 
                                        "D239N_1", "D239N_2", "D239N_3",
                                        "K637E_1", "K637E_2", "K637E_3")))

write_csv(df, "D:/Projects/MAB_project/Delphi_Force/csv/results_residue_formatted.csv")

df_avg <- df %>% 
  group_by(Simulation, Residue_ID, ID, Genotype) %>% 
  reframe(binding_force = mean(binding_force),
          G = mean(G))

df_avg <- df_avg %>% 
  filter(!(Residue_ID == "637_LYS" & Genotype == "K637E"))

df_avg %>% 
  filter(ID == 637)

#Pull out just residues that are changing by more than 0.01
df_important <- df_avg %>% 
  group_by(Residue_ID) %>% 
  mutate(Gmax = max(G)) %>%
  mutate(Gmin = min(G)) %>%
  filter(Gmin < -0.01 | Gmax > 0.01) #This cutoff is arbitrary 

print(df_important$ID %>% unique())

write_csv(df_important, "D:/Projects/MAB_project/Delphi_Force/csv/results_residue_important.csv")

# Graph energy at each residue  ------------------------------------------
#Look at Loop 2  
df_loop2 <- df_avg %>% 
  #filter(ID %in% c(633,635,637,639,640)) %>% 
  filter(ID>621 & ID<647)

p1 <- ggplot()+
  geom_col(data = df_loop2, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:26), linetype="dotted")+
  
  stat_compare_means(
    data = df_loop2 %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_loop2$binding_force) * 1.1)+
  
  labs(title = "Loop 2 ", x = "Residue", y = "Binding Force (nN)") +
  theme_pubr() +
  guides(fill=guide_legend(nrow=3))+
  theme(text = element_text(size=18, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/Loop2_binding_force.png",
       plot = p1, width = 12, height = 6, dpi = 500)

#Look at Loop 3  
df_loop3 <- df_avg %>% 
  filter(ID>566 & ID<578)

p2 <- ggplot()+
  geom_col(data = df_loop3, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:12), linetype="dotted")+
  
  stat_compare_means(
    data = df_loop3 %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_loop3$binding_force) * 1.1)+
  
  labs(title = "Loop 3", x = "Residue", y = "Binding Force (nN)")+
  guides(fill=guide_legend(nrow=3))+
  theme_pubr() +
  theme(text = element_text(size=10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))
  #ylim(-.045,.01)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/Loop3_binding_energy.png",
       plot = p2, width = 9, height = 6, dpi = 500)

#Look at Upper 50kDa
df_loop4 <- df_avg %>% 
  filter(ID>361 & ID<378)

p3 <- ggplot()+
  geom_col(data = df_loop4, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:17), linetype="dotted")+
  
  stat_compare_means(
    data = df_loop4 %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_loop4$binding_force) * 1.1)+
  
  labs(title = "Loop 4", x = "Residue", y = "Binding Force (nN)") +
  guides(fill=guide_legend(nrow=3))+
  theme_pubr() +
  theme(text = element_text(size=10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/Loop4_binding_energy.png",
       plot = p3, width = 9, height = 6, dpi = 500)

#Cardiomyopathy loop
df_cm_loop <- df_avg %>% 
  filter(ID>400 & ID<417)

p4 <- ggplot()+
  geom_col(data = df_cm_loop, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:17), linetype="dotted")+
  
  stat_compare_means(
    data = df_cm_loop %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_cm_loop$binding_force) * 1.1)+
  
  labs(title = "Cardiomyopathy Loop", x = "Residue", y = "Binding Force (nN)") +
  guides(fill=guide_legend(nrow=3))+
  theme_pubr() +
  theme(text = element_text(size=10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))
  #ylim(-.045,.01)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/Cardiomyopathy_Loop_binding_energy.png",
       plot = p4,width = 9, height = 6, dpi = 500)


#Lower 50 Tip
df_lowwer50_tip <- df_avg %>% 
  filter(ID>538 & ID<545)

p5 <- ggplot()+
  geom_col(data = df_lowwer50_tip, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:7), linetype="dotted")+
  
  stat_compare_means(
    data = df_lowwer50_tip %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_lowwer50_tip$binding_force) * 1.1)+
  
  labs(title = "Lower 50 kDa Tip", x = "Residue", y = "Binding Force (nN)") +
  guides(fill=guide_legend(nrow=3))+
  theme_pubr() +
  theme(text = element_text(size=10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))
  # ylim(-.045,.01)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/Lower50_tip_binding_energy.png",
       plot = p5, width = 9, height = 6, dpi = 500)

#tail
df_tail <- df_avg %>% 
  filter(ID>765 & ID<811)

p6 <- ggplot()+
  geom_col(data = df_tail, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:45), linetype="dotted")+
  
  stat_compare_means(
    data = df_tail %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_tail$binding_force) * 1.1)+
  
  labs(title = "Tail", x = "Residue", y = "Binding Force (nN)") +
  guides(fill=guide_legend(nrow=3))+
  theme_pubr() +
  theme(text = element_text(size=10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))
  #ylim(-0.5,0.5)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/Tail_energy.png",
       plot = p6, width = 23, height = 6, dpi = 500)

#helix-loop-helix
df_HLH <- df_avg %>% 
  filter(ID>520 & ID<556)

p12 <- ggplot()+
  geom_col(data = df_HLH, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:36), linetype="dotted")+
  
  stat_compare_means(
    data = df_HLH %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_HLH$binding_force) * 1.1)+
  
  labs(title = "Helix-Loop-Helix Motif", x = "Residue", y = "Binding Force (nN)") +
  guides(fill=guide_legend(nrow=3))+
  theme_pubr() +
  theme(text = element_text(size=10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))
  #ylim(-0.5,0.5)
  
  ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/HLH_energy.png",
         plot = p12, width = 15, height = 6, dpi = 500)

#S1 Loop
df_S1 <- df_avg %>% 
  filter(ID>232 & ID<244)

p13 <- ggplot()+
  geom_col(data = df_S1, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:12), linetype="dotted")+
  
  stat_compare_means(
    data = df_S1 %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_S1$binding_force) * 1.1)+
  
  labs(title = "S1", x = "Residue", y = "Binding Force (nN)") +
  guides(fill=guide_legend(nrow=3))+
  theme_pubr() +
  theme(text = element_text(size=10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))
  #ylim(-0.5,0.5)
  
  ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/S1_energy.png",
         plot = p13, width = 15, height = 6, dpi = 500)
  

#SH1 Loop
df_SH1 <- df_avg %>% 
  filter(ID>674 & ID<685)

p14 <- ggplot()+
  geom_col(data = df_SH1, 
           aes(x = Residue_ID, y = binding_force, fill = Simulation),
           position = "dodge")+
  geom_vline(xintercept = c(1.5:11), linetype="dotted")+
  
  stat_compare_means(
    data = df_SH1 %>%
      group_by(Residue_ID) %>%
      filter(n_distinct(Genotype) >= 2),
    
    aes(x = Residue_ID, y = binding_force, group = Genotype),
    method = "anova",
    label = "p.signif",
    hide.ns = TRUE,
    size = 4,
    label.y = max(df_SH1$binding_force) * 1.1)+
  
  labs(title = "SH1", x = "Residue", y = "Binding Force (nN)") +
  guides(fill=guide_legend(nrow=3))+
  theme_pubr() +
  theme(text = element_text(size=10, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c('#000000', '#343436', '#545459', 'lightgreen', 'green',  "darkgreen", 'lightblue', 'blue', 'darkblue'))
  #ylim(-0.5,0.5)
  
  ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_energy_residue/SH1_energy.png",
         plot = p14, width = 15, height = 6, dpi = 500)

# Graph all important residues  -------------------------------------------
df_important_avgGt <- df_important %>% 
  group_by(Genotype, Residue_ID, ID) %>% 
  reframe(binding_force = mean(binding_force))


#Graph everybody
p7 <- ggplot()+
  geom_col(data = df_important_avgGt,
           aes(x = Residue_ID, y = binding_force, fill = Genotype),
           position = "dodge")+
  geom_vline(xintercept = c(.5:37), linetype="dotted", alpha = .5)+

  #Upper 50 kDa
  annotate("rect",
           xmin = 0.5, xmax = 2.5,
           ymin = -7, ymax = 7,
           alpha = .3,
           fill = "lightblue")+
  annotate("text",
           x=2,
           y=4,
           label= "Up 50 kDa",
           fontface = 2,
           size = 7) +

  #Lower 50 kDa
  annotate("rect",
           xmin = c(2.5,24.5,34.5), xmax = c(20.5,25.5,36.5),
           ymin = -7, ymax = 7,
           alpha = .3,
           fill = "blue")+
  annotate("text",
           x=12,
           y=4,
           label= "Low 50 kDa",
           fontface = 2,
           size = 7) +

  #Loop 3
  annotate("rect",
           xmin = 20.5, xmax = 24.5,
           ymin = -7, ymax = 7,
           alpha = .2,
           fill = "blue")+
  annotate("text",
           x=(22.5),
           y=-4,
           label= "Loop 3",
           fontface = 2,
           size = 7) +

  #Strut
  annotate("rect",
           xmin = 25.5, xmax = 26.5,
           ymin = -7, ymax = 7,
           alpha = .15,
           fill = "green")+
  annotate("text",
           x=(26),
           y=4,
           label= "Strut",
           fontface = 2,
           size = 7) +
  #Loop 2
  annotate("rect",
           xmin = 26.5, xmax = 34.5,
           ymin = -7, ymax = 7,
           alpha = .2,
           fill = "blue")+
  annotate("text",
           x=(30.5),
           y=-4,
           label= "Loop 2",
           fontface = 2,
           size = 7) +
  
  geom_col(data = df_important_avgGt,
           aes(x = Residue_ID, y = binding_force, fill = Genotype),
           position = "dodge")+
  
  scale_x_discrete(drop = TRUE)+
  
  labs(title = "Important Residues: Force", x = "Residue", y = "Binding Force (nN)") +
  theme_pubr() +
  theme(text = element_text(size=18, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c(wt_color, d239n_color, k637e_color))
  #ylim(-7,7)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force_residue/Important_Residues_color.png",
       plot = p7, width = 19, height = 6, dpi = 500)

#All Energy no color
p8 <- ggplot()+
  geom_col(data = df_important_avgGt, 
           aes(x = Residue_ID, y = binding_force, fill = Genotype),
           position = "dodge")+
  geom_vline(xintercept = c(.5:65), linetype="dotted", alpha = .5)+
  
  labs(title = "Important Residues: Force", x = "Residue", y = "Binding Force (nN)") +
  theme_pubr() +
  theme(text = element_text(size=18, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c(wt_color, d239n_color, k637e_color))+
  ylim(-7,7)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force_residue/Important_Residues.png",
       plot = p8, width = 19, height = 6, dpi = 500)

df_important_avgGt_wt <- df_important_avgGt %>%
  filter(Genotype == "WT") %>%
  rename(binding_force_wt = binding_force) %>%
  select(ID, binding_force_wt)

df_important_avgGt_change <- left_join(df_important_avgGt, df_important_avgGt_wt, by = 'ID')

df_important_avgGt_change %>% filter(ID == 637)

df_important_avgGt_change <- df_important_avgGt_change %>%
  mutate(percent_change = (binding_force-binding_force_wt)/binding_force_wt*100)



#Graph everybody % change
p9 <- ggplot()+
  geom_col(data = df_important_avgGt_change %>% filter(Genotype != "WT"), 
           aes(x = Residue_ID, y = percent_change, fill = Genotype),
           position = "dodge")+
  geom_vline(xintercept = c(.5:37), linetype="dotted", alpha = .5)+
  
  #Upper 50 kDa
  annotate("rect",
           xmin = 0.5, xmax = 2.5,
           ymin = -202, ymax = 75,
           alpha = .3,
           fill = "lightblue")+
  annotate("text",
           x=2,
           y=40,
           label= "Up 50 kDa",
           fontface = 2,
           size = 7) +
  
  #Lower 50 kDa
  annotate("rect",
           xmin = c(2.5,24.5,34.5), xmax = c(20.5,25.5,36.5),
           ymin = -202, ymax = 75,
           alpha = .3,
           fill = "blue")+
  annotate("text",
           x=12,
           y=40,
           label= "Low 50 kDa",
           fontface = 2,
           size = 7) +
  
  #Loop 3
  annotate("rect",
           xmin = 20.5, xmax = 24.5,
           ymin = -202, ymax = 75,
           alpha = .2,
           fill = "blue")+
  annotate("text",
           x=(22.5),
           y=40,
           label= "Loop 3",
           fontface = 2,
           size = 7) +
  
  #Strut
  annotate("rect",
           xmin = 25.5, xmax = 26.5,
           ymin = -202, ymax = 75,
           alpha = .15,
           fill = "green")+
  annotate("text",
           x=(26),
           y=40,
           label= "Strut",
           fontface = 2,
           size = 7) +
  #Loop 2
  annotate("rect",
           xmin = 26.5, xmax = 34.5,
           ymin = -202, ymax = 75,
           alpha = .2,
           fill = "blue")+
  annotate("text",
           x=(30.5),
           y=40,
           label= "Loop 2",
           fontface = 2,
           size = 7) +
  
  geom_col(data = df_important_avgGt_change %>% filter(Genotype != "WT"), 
           aes(x = Residue_ID, y = percent_change, fill = Genotype),
           position = "dodge")+
  
  labs(title = "Important Residues: % Change", x = "Residue", y = "% Change from WT") +
  theme_pubr() +
  theme(text = element_text(size=18, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.line.y.right = element_blank())+
  scale_fill_manual(values = c(k637e_color, d239n_color))+
  scale_y_break(breaks = c(-75, -190), scales = 10)+
  scale_y_continuous(breaks = c(-200, -100, -75, -50, 0, 50, 75))

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force_residue/Percent_change_color.png",
       plot = p9, width = 19, height = 6, dpi = 500)

#Graph everybody % change
p10 <- ggplot()+
  geom_col(data = df_important_avgGt_change %>% filter(Genotype != "WT"), 
           aes(x = Residue_ID, y = percent_change, fill = Genotype),
           position = "dodge")+
  annotate("rect",
           xmin = 26.5, xmax = 34.5,
           ymin = -202, ymax = 75,
           alpha = .2,
           fill = "white")+
  geom_vline(xintercept = c(.5:65), linetype="dotted", alpha = .5)+
  
  labs(title = "Important Residues: % Change", x = "Residue", y = "Percent Chagne (%)") +
  theme_pubr() +
  theme(text = element_text(size=18, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.line.y.right = element_blank())+
  scale_fill_manual(values = c(k637e_color, d239n_color))+
  scale_y_break(breaks = c(-75, -190), scales = 10)+
  scale_y_continuous(breaks = c(-200, -100, -75, -50, 0, 50, 75))

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force_residue/Percent_change.png",
       plot = p10, width = 19, height = 6, dpi = 500)



#Graph Loop 2 Lys and and 637
df_loop2_lys <- df_loop2 %>% 
  filter(grepl("LYS", Residue_ID) |
           grepl("637", Residue_ID))

df_loop2_lys_stats <- df_loop2_lys %>%
  group_by(Residue_ID) %>%
  filter(n_distinct(Genotype) > 1)


p11 <- ggplot()+
  geom_point(data = df_loop2_lys,
             aes(x = Genotype, y = binding_force, color = Genotype),
             size = 3,
             shape = 16,
             position = position_jitterdodge(jitter.width = .4,
                                             dodge.width = .6,)) +
  
  stat_summary(data = df_loop2_lys,
               fun = mean,
               geom = "bar",
               width = .4,
               aes(x = Genotype, y = binding_force, fill = Genotype),
               linewidth = 1, color = "black", alpha = 0.2, position = position_dodge(.6))+
  
  stat_summary(data = df_loop2_lys,
               aes(x = Genotype, y = binding_force, group = Genotype),
               fun.data = mean_se,
               geom = "errorbar",
               position = position_dodge(.6),
               width = 0.1, color = "black", linewidth = 1) +
  
  
  stat_compare_means(
              data = df_loop2_lys_stats,
              aes(x = Genotype, y = binding_force, group = Genotype),
              method = "anova",
              size = 5,label.y = 11, label.x = 1,
              label = "p.format",
              show.legend = F,
              fontface = "bold")+
  
  labs(title = "Loop 2 Lys", x = "Residue", y = "Binding Force (nN)") +
  theme_pubr() +
  theme(text = element_text(size=18, face = "bold"), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c(wt_color, d239n_color, k637e_color))+
  scale_color_manual(values = c(wt_color, d239n_color, k637e_color))+
  facet_wrap(~Residue_ID, nrow = 1)+
  ylim(-12,12)

ggsave("D:/Projects/MAB_project/Delphi_Force/results/plots_force_residue/loop2_lys.png",
       plot = p11, width = 10, height = 6, dpi = 500)

# Find the max residue of each sim and genotype
df_minmax_sim <- df %>%
  group_by(Genotype, sim, Sim_ID) %>%
  summarise(
    min_force    = min(binding_force, na.rm = TRUE),
    min_time     = time[which.min(binding_force)],
    min_residue  = Residue_ID[which.min(binding_force)],
    
    max_force    = max(binding_force, na.rm = TRUE),
    max_time     = time[which.max(binding_force)],
    max_residue  = Residue_ID[which.max(binding_force)],
    
    .groups = "drop"
  )
