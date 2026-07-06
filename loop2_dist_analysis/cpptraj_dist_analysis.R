# Minimal debugging script for cpptraj distance analysis
# Run this line by line to see where any errors occur

# Load libraries ----------------------------------------------------------
library(tidyverse)
library(ggpubr)

print("Libraries loaded successfully!")

# Test reading ONE file first ---------------------------------------------
print("Testing file reading...")

# Try to read the first WT file
wt1 <- read_table("WT_sim1_vector.dat", skip = 1, 
                  col_names = c("Frame", "VectorX", "VectorY", "VectorZ"), 
                  show_col_types = FALSE)

print("File read successfully!")
print(paste("Number of rows:", nrow(wt1)))
print("First few rows:")
print(head(wt1))

# Process the data
wt1$frame <- as.numeric(wt1$Frame)
wt1$x <- as.numeric(wt1$VectorX)
wt1$y <- as.numeric(wt1$VectorY)
wt1$z <- as.numeric(wt1$VectorZ)
wt1$time <- 1:nrow(wt1)
wt1$sim <- 1
wt1$Genotype <- "Wt"

print("Data processed!")
print(head(wt1))

# Calculate distance for this one file
actin_nterm_x <- -57.33325
actin_nterm_y <- 4.03525
actin_nterm_z <- 563.8495

wt1$dist <- sqrt((wt1$x - actin_nterm_x)^2 + 
                 (wt1$y - actin_nterm_y)^2 + 
                 (wt1$z - actin_nterm_z)^2)

print("Distance calculated!")
print(paste("Mean distance:", mean(wt1$dist)))
print(paste("Min distance:", min(wt1$dist)))
print(paste("Max distance:", max(wt1$dist)))

# Make a simple plot
plot(wt1$time, wt1$dist, type = "l", 
     main = "WT Simulation 1",
     xlab = "Time (ns)", 
     ylab = "Distance (Å)")

print("\nIf you see this message and a plot, the script is working!")
print("Now you can run the full script.")