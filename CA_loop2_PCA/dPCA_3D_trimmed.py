#!/usr/bin/env python
from numpy import *
import numpy as np
import sys
import csv
from sklearn import decomposition
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc, font_manager
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from scipy.stats import gaussian_kde
import pandas as pd
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###########################################
#  Read in pickled data
###########################################
# Read CSV instead of pickle
Q_reduced = pd.read_csv("pca_output/Q_reduced_trimmed.csv").values
y = pd.read_csv("pca_output/y_trimmed.csv")["sim_id"].values
dist = pd.read_csv("pca_output/dist_trimmed.csv")["dist"].values

###########################################
#  3D PCA PLOT - Genotype
###########################################
color_dict = {0: 'lightgrey', 1: 'grey', 2: 'black',
              3: 'lightgreen', 4: 'green', 5: 'darkgreen',
              6: 'lightblue', 7: 'blue', 8: 'darkblue'}

sim_names = {
    0:"WT_1", 1:"WT_2", 2:"WT_3",
    3:"D239N_1", 4:"D239N_2", 5:"D239N_3",
    6:"K637E_1", 7:"K637E_2", 8:"K637E_3"
}

fig_sim = go.Figure()

for sim_val in sorted(sim_names.keys()):
    mask = (y == sim_val)
    if mask.sum() == 0:
        continue

    fig_sim.add_trace(go.Scatter3d(
        x=Q_reduced[mask, 0],
        y=Q_reduced[mask, 1],
        z=Q_reduced[mask, 2],
        mode='markers',
        name=sim_names[sim_val],
        marker=dict(color=color_dict[sim_val], size=3, opacity=1)
    ))

fig_sim.update_layout(
    title="PCA Plot (colored by Simulation)",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
)

fig_sim.write_html("pca_output/pca_3d_plot_sim.html")
fig_sim.show()

###########################################
#  3D PCA PLOT - Dist
###########################################
fig_dist = go.Figure()

fig_dist.add_trace(go.Scatter3d(
    x=Q_reduced[:, 0],
    y=Q_reduced[:, 1],
    z=Q_reduced[:, 2],
    mode='markers',
    showlegend=False,
    marker=dict(
        color=dist,
        colorscale=[[0, "purple"], [1, "yellow"]],  # <-- matches R low/high
        colorbar=dict(title="Distance (Å)"),
        size=3,
        opacity=1
    )
))

fig_dist.update_layout(
    title="PCA Plot (colored by Distance)",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
)

fig_dist.write_html("pca_output/pca_3d_plot_dist.html")
fig_dist.show()

###########################################
#  LOAD FORCE/DISTANCE DATA
###########################################
force_df = pd.read_csv(r"D:\Projects\MAB_project\CA_Loop2_PCA_Analysis\kalen_csv\csv\results.csv")
y = force_df['binding_energy'].values * -41.14  # convert to nN, flip to positive

###########################################
#  3D PCA PLOT - Force
###########################################
fig_dist = go.Figure()

fig_dist.add_trace(go.Scatter3d(
    x=Q_reduced[:, 0],
    y=Q_reduced[:, 1],
    z=Q_reduced[:, 2],
    mode='markers',
    showlegend=False,
    marker=dict(
        color=y,
        colorscale=[[0, "purple"], [1, "yellow"]],
        colorbar=dict(title="Force (nN)"),
        size=3,
        opacity=1
    )
))

fig_dist.update_layout(
    title="PCA Plot (colored by Binding Force)",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
)

fig_dist.write_html("pca_output/pca_3d_plot_force.html")
fig_dist.show()