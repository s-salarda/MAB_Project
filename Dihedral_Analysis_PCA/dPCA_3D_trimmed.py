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
from scipy.ndimage import generate_binary_structure, binary_erosion
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

###########################################
#  Read in pickled data
###########################################

# Open the pickled file in read binary mode
with open("pca_output/Q_reduced_trimmed.data", "rb") as f:
  # Use pickle.load to deserialize the object from the file
  Q_reduced = pickle.load(f)

with open("pca_output/y_trimmed.data", "rb") as f:
  # Use pickle.load to deserialize the object from the file
  y = pickle.load(f)

###########################################
#  3D PCA PLOT
###########################################

# Define a color dictionary for genotypes
# Custom color dictionary for genotypes
color_dict = {0: "lightgrey", 1: "grey", 2: "black", 
            3: "lightgreen", 4: "green", 5: "darkgreen", 
            6: "lightblue", 7: "blue", 8: "darkblue"}  # Customize colors as needed

# Create a Scatter3d trace with color based on genotype
trace = go.Scatter3d(
    x=Q_reduced[:, 0],  # Select first principal component for x-axis
    y=Q_reduced[:, 1],  # Select second principal component for y-axis
    z=Q_reduced[:, 2],  # Select third principal component for z-axis (optional)
    mode='markers',  # Set mode to display individual data points
    marker=dict(
        color=[color_dict[val] for val in y],  # Set marker color based on genotype
        size=3,  # Adjust marker size (optional)
        opacity=1  # Adjust marker transparency (optional)
    )
)

GENOTYPE_LABELS = [
    'WT_1','WT_2','WT_3','D239N_1','D239N_2','D239N_3','K637E_1','K637E_2','K637E_3'
]

import numpy as np
import os
import plotly.graph_objects as go

import numpy as np
import os
import plotly.graph_objects as go

def plot_3d_pca(Q_reduced, y, output_dir, pca=None, trim_frames=0):
    N = Q_reduced.shape[0]
    global_idx = np.arange(N, dtype=int)

    colors = [color_dict[int(val)] for val in y]
    labels = [GENOTYPE_LABELS[int(val)] for val in y]  # WT_1..K637E_3 [1](https://onedrive.live.com/?id=b253dbf6-6731-4a0c-b8cb-cd2e00fbeb7c&cid=15d96f8dabc1993e&web=1)

    # Frame counter that resets for each value in y (i.e., each replicate label)
    counts = {}
    frame_in_rep = np.empty(N, dtype=int)
    for i, v in enumerate(y):
        v = int(v)
        frame_in_rep[i] = counts.get(v, 0)
        counts[v] = frame_in_rep[i] + 1
    frame_in_rep = frame_in_rep + trim_frames

    # customdata: [replicate_frame, global_index]
    customdata = np.column_stack([frame_in_rep, global_idx])

    # Axis labels with variance (optional)
    if pca is not None and hasattr(pca, "explained_variance_ratio_") and len(pca.explained_variance_ratio_) >= 3:
        v = pca.explained_variance_ratio_ * 100.0
        xlab = f"PC1 ({v[0]:.2f}%)"
        ylab = f"PC2 ({v[1]:.2f}%)"
        zlab = f"PC3 ({v[2]:.2f}%)"
    else:
        xlab, ylab, zlab = "PC1", "PC2", "PC3"

    fig = go.Figure(data=[go.Scatter3d(
        x=Q_reduced[:, 0],
        y=Q_reduced[:, 1],
        z=Q_reduced[:, 2],
        mode='markers',
        marker=dict(color=colors, size=3),
        text=labels,
        customdata=customdata,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Frame (in replicate): %{customdata[0]}<br>"
            "Global index: %{customdata[1]}<br>"
            "PC1: %{x:.3f}<br>"
            "PC2: %{y:.3f}<br>"
            "PC3: %{z:.3f}<br>"
            "<extra></extra>"
        )
    )])

    fig.update_layout(
        title="PCA 3D Visualization",
        scene=dict(
            xaxis_title=xlab,
            yaxis_title=ylab,
            zaxis_title=zlab
        ),
        width=1000,
        height=800
    )

    outfile = os.path.join(output_dir, "PCA_3D_interactive.html")
    fig.write_html(outfile)
    return outfile

# Display the plot
plot_3d_pca(Q_reduced, y, "pca_output")
