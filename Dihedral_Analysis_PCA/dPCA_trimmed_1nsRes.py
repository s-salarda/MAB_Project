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
import pickle


###########################################
#  DEF FUNCTIONS 
###########################################

def all_combinations(data):
	return combinations(data, 2)  # Change 2 to desired combination size

def importResPhiPsi(resnum, root):    
	phi = np.loadtxt(root + str(resnum) + '_phi.dat', usecols=(1,), dtype=float)
	psi = np.loadtxt(root + str(resnum) + '_psi.dat', usecols=(1,), dtype=float)
	return phi, psi

def importMD(md_dir):
	Q = np.zeros((numframes, 4*numres))
	for i in range(0, numres):
		phi, psi = importResPhiPsi(resrange[i],md_dir)

		phi = phi[::1] #This ::100 slicing notation means "start from the beginning (0th index), take every 100th element, and continue until the end".
		psi = psi[::1] # 100: Start at the 100th frame

		j = 4 * i
		Q[:, j] = np.cos(phi * np.pi / 180.)
		Q[:, j + 1] = np.sin(phi * np.pi / 180.)
		Q[:, j + 2] = np.cos(psi * np.pi / 180.)
		Q[:, j + 3] = np.sin(psi * np.pi / 180.)
	return Q

def plot_pc_vs_pc_logarithmic(Q_full_reduced, x_pc, y_pc, pc_vs_pc_name):
	fig = plt.figure()
	ax = plt.gca()
	[H, xedge, yedge, im] = plt.hist2d(Q_full_reduced[:, x_pc], Q_full_reduced[:, y_pc], bins=100)

	prefix = "PC" + str(x_pc + 1) + "_vs_PC" + str(y_pc + 1)
	plt.xlabel('PC' + str(x_pc + 1) + ' [ ' + str(pca.explained_variance_ratio_[x_pc] * 100)[0:4] + ' % variance ]')
	plt.ylabel('PC' + str(y_pc + 1) + ' [ ' + str(pca.explained_variance_ratio_[y_pc] * 100)[0:4] + ' % variance ]')
	ax.xaxis.set_tick_params(width=2, length=7)
	ax.yaxis.set_tick_params(width=2, length=7)
	plt.title("PC" + str(x_pc + 1) + " vs PC" + str(y_pc + 1))
	plt.savefig(pc_vs_pc_name + '.png', format='png', dpi=400, bbox_inches='tight', edgecolor='none')
	plt.clf()
	plt.cla()
	plt.close()

def plot_pc_logarithmic(Q_reduced, x_pc, pc_name):

	fig = plt.figure()
	ax = plt.gca()
	n, bins, patches = plt.hist(Q_reduced[:,x_pc], bins=100)#, cmap=plt.get_cmap('gnuplot'), norm=LogNorm()) # may need to optimize number of bins as function of pc number
	#plt.set_xlabel('PC' + str(x_pc + 1) + ' [ ' + str(pca.explained_variance_ratio_[x_pc]*100)[0:4] + ' % variance ]')
	#plt.set_ylabel('Counts')
	#f.subplots_adjust(bottom=0.2)
	#fontProperties = {'family': 'sans-serif', 'weight': 'normal', 'size': 10}
	#ticks_font = font_manager.FontProperties(style='normal', size=14, weight='bold', stretch='normal')
	#rc('font', **fontProperties)
	plt.title("PC " + str(x_pc + 1))
	plt.savefig(pc_name + '.png', format='png', dpi=400, bbox_inches='tight', edgecolor='none')
	plt.clf()
	plt.cla()
	plt.close()
	#return [n, bins, patches]

def plot_explained_variance(pca):
	fig = plt.figure(1, figsize=(4, 3))
	ax = fig.add_subplot(111)
	ax.bar(range(1, pca.n_components_ + 1), 100.*pca.explained_variance_ratio_, linewidth=1, color='#1E90FF', align="center")
	plt.axis('tight')
	plt.xlabel('Principal Components')
	plt.ylabel('Variance Explained (%)')
	ax.xaxis.set_tick_params(width=1, length=7)
	ax.yaxis.set_tick_params(width=1, length=7)
	fig.subplots_adjust(bottom=0.2)
	fontProperties = {'family': 'sans-serif', 'weight': 'normal', 'size': 14}
	ticks_font = font_manager.FontProperties(style='normal', size=14, weight='bold', stretch='normal')
	rc('font', **fontProperties)
	prefix = "variance"
	plt.savefig(prefix + '.png', format='png', dpi=400, bbox_inches='tight', edgecolor='none')
	
	cumulative = np.cumsum(pca.explained_variance_ratio_)
	'''
	with open(prefix + '.txt', 'wb') as f:
		f.write("Number of Components: " + str(pca.n_components_) + '\n')
		for i in range(0, pca.n_components_):
			f.write(str(i+1) + '\t' + str(100.*pca.explained_variance_ratio_[i]) + '\t' + str(cumulative[i]) + '\n')
			'''
	plt.clf()
	plt.cla()
	plt.close()


def plot_weights(resnums, pca, n_pc):
	
	for i in range(0, n_pc):
		#f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
		fig = plt.figure()
		ax = plt.gca()
		ax.bar(range(0, 4 * len(resnums)), np.absolute(pca.components_[i, :]), color='#1E90FF', align="center")
		ax.set_title('PC' + str(i+1) + ' [ ' + str(pca.explained_variance_ratio_[i] * 100)[0:4] + ' % variance ]')

		labels = [str(x) for x in resnums]
		ticks = np.linspace(0, 4*len(resnums), num=4, endpoint=False)
		plt.axis('tight')
		#plt.xticks(ticks, labels)
		ax.set_ylabel('Weights on PC Vectors')
		prefix = 'weights_pc' + str(i+1)
		plt.savefig(prefix + '.png', format='png', dpi=200, bbox_inches='tight', edgecolor='none')
		plt.clf()
		plt.cla()
		plt.close()

def plot_pc_vs_pc_colored_by_genotype(Q, y, x_pc, y_pc, pc_vs_pc_name):
  """
  This function creates a PCA plot with points colored based on genotypes.

  Args:
      Q: A 2D NumPy array representing the data after PCA.
      y: A 1D NumPy array representing the genotypes (1 or 0).
      x_pc: An integer specifying the index of the first PC for the x-axis.
      y_pc: An integer specifying the index of the second PC for the y-axis.
      pc_vs_pc_name: A string used as the base filename when saving the plot.
  """

  fig, ax = plt.subplots()  # Create figure and axes together

  # Custom color dictionary for genotypes
  color_dict = {0: "lightgrey" , 1: "darkgrey", 2: "black", 
                3: "lightgreen", 4: "green", 5: "darkgreen", 
                6: "lightblue", 7: "blue", 8: "darkblue"}  # Customize colors as needed

  # Scatter plot with color based on genotype using dictionary lookup
  ax.scatter(Q[:, x_pc], Q[:, y_pc], 
  	c=[color_dict[val] for val in y],
  	edgecolors=[color_dict[val] for val in y],
  	alpha=1, s=6) #, marker = "o"

  prefix = "PC" + str(x_pc + 1) + "_vs_PC" + str(y_pc + 1)
  plt.xlabel('PC' + str(x_pc + 1) + ' [ ' + str(pca.explained_variance_ratio_[x_pc] * 100)[0:4] + ' % variance ]', fontweight='bold', fontsize=20)
  plt.ylabel('PC' + str(y_pc + 1) + ' [ ' + str(pca.explained_variance_ratio_[y_pc] * 100)[0:4] + ' % variance ]', fontweight='bold', fontsize=20)
  ax.xaxis.set_tick_params(width=2, length=7)
  ax.yaxis.set_tick_params(width=2, length=7)
  plt.xticks(fontsize=20, fontweight='bold')
  plt.yticks(fontsize=20, fontweight='bold')
  plt.title("PC" + str(x_pc + 1) + " vs PC" + str(y_pc + 1), fontweight='bold', fontsize=20)
  plt.savefig(pc_vs_pc_name + '.png', format='png', dpi=400, bbox_inches='tight', edgecolor='none')
  plt.legend()
  plt.clf()
  plt.cla()
  plt.close()

###########################################
#  SET UP VARIABLES
###########################################
numframes = 500  # assuming 500ns simulation with 1ns sampling

#We want the residues from loop 2 but we want to exclude Gly residues because they are highly variable Glys are 626,634,636,641 
resrange = np.concatenate((np.arange(621, 626), np.arange(627, 634), np.arange(635, 636), np.arange(637, 641), np.arange(642, 647))) 
# gives us all of loop 2 621,647

numres = len(resrange)
root_dirs_wt1 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_wt_1\Loop_II_"
root_dirs_wt2 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_wt_2\Loop_II_"
root_dirs_wt3 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_wt_3\Loop_II_"
root_dirs_d239n1 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_d239n_1\Loop_II_"
root_dirs_d239n2 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_d239n_2\Loop_II_"
root_dirs_d239n3 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_d239n_3\Loop_II_"
root_dirs_k637e1 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_k637e_1\Loop_II_"
root_dirs_k637e2 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_k637e_2\Loop_II_"
root_dirs_k637e3 = r"D:\Projects\MAB_project\Dihedral_Analysis_PCA_Fixed\results\dihedrals_k637e_3\Loop_II_"

print("Residu Range")
print(resrange)
print("numres =")
print(numres)
print("numframes =")
print(numframes)

print("Importing data...")

###########################################
#  IMPORT MD DATASET
###########################################
# Shape of Q should be (number of time points, 4* number of residues)
# 4*num residues accounts for sin/cos components for each residue
# There will be special cases for any N-/C-terminal residues that are not implemented yet

# Q_wt1 = importMD(root_dirs_wt1)[100:]
# Q_wt2 = importMD(root_dirs_wt2)[100:]
# Q_wt3 = importMD(root_dirs_wt3)[100:]
# Q_d239n1 = importMD(root_dirs_d239n1)[100:]
# Q_d239n2 = importMD(root_dirs_d239n2)[100:]
# Q_d239n3 = importMD(root_dirs_d239n3)[100:]
# Q_k637e1 = importMD(root_dirs_k637e1)[100:]
# Q_k637e2 = importMD(root_dirs_k637e2)[100:]
# Q_k637e3 = importMD(root_dirs_k637e3)[100:]

# For every single frame
Q_wt1 = importMD(root_dirs_wt1)
Q_wt2 = importMD(root_dirs_wt2)
Q_wt3 = importMD(root_dirs_wt3)
Q_d239n1 = importMD(root_dirs_d239n1)
Q_d239n2 = importMD(root_dirs_d239n2)
Q_d239n3 = importMD(root_dirs_d239n3)
Q_k637e1 = importMD(root_dirs_k637e1)
Q_k637e2 = importMD(root_dirs_k637e2)
Q_k637e3 = importMD(root_dirs_k637e3)

#print("Shape of Q_wt1:")
#print(Q_wt1.shape)
#print(Q_wt1)

Q = np.concatenate((Q_wt1, Q_wt2, Q_wt3, Q_d239n1, Q_d239n2, Q_d239n3, Q_k637e1, Q_k637e2, Q_k637e3))
print("Shape of Q:")
print(Q.shape)

# y will be the genotype variable 
# 0-2 will be 'WT', 3-5 will be 'E525K', 6-8 will be 'V606M'
# y = np.concatenate((np.zeros(numframes-100), np.ones(numframes-100), np.ones(numframes-100)*2, np.ones(numframes-100)*3, np.ones(numframes-100)*4,np.ones(numframes-100)*5, np.ones(numframes-100)*6, np.ones(numframes-100)*7, np.ones(numframes-100)*8))

# For single frames
y = np.concatenate((np.zeros(numframes), np.ones(numframes), np.ones(numframes)*2, np.ones(numframes)*3, np.ones(numframes)*4, np.ones(numframes)*5, np.ones(numframes)*6, np.ones(numframes)*7, np.ones(numframes)*8))
#print("Shape of y:")
#print(y.shape)
#print(y)

print("Performing PCA...")

###########################################
#  PERFORM PCA
###########################################
pca = decomposition.PCA(n_components=4*numres)
pca.fit(Q)

print("Shape of Q:")
print(Q.shape)
print(Q)



print("Transforming coordinates...")
Q_reduced = pca.transform(Q)

# IF PCA TAKES A WHILE TO RUN YOU CAN PICKLE THE PROJECTION
# Open a file for writing in binary mode
print("Pickling Data...")
with open("pca_output/Q_reduced_trimmed.data", "wb") as f:
  pickle.dump(Q_reduced, f)

with open("pca_output/y_trimmed.data", "wb") as f:
  pickle.dump(y, f)

print("Done Pickling Data...")
#End of the PCA Calculations, everything from here on is just graphing data in different ways 


pcs_to_compaire = range(5)
for combo in all_combinations(pcs_to_compaire):
	plot_pc_vs_pc_colored_by_genotype(Q_reduced, y, combo[0], combo[1], "PC " + str(combo[0]+1) + " vs. PC " + str(combo[1]+1))



###########################################
#  SCREE PLOT
###########################################
plot_explained_variance(pca)

###########################################
#  EVAL PLOTS 
###########################################
num_components = 4*numres # Total number of evectors, this will for loop plot weights on all of your eigenvectors
plot_weights(resrange, pca, num_components)

###########################################
# SINGLE EVECT PROJECTION
###########################################
evect_number = 0 # the particular pc axis you want to project to. Possible options are 0 -> (4*numres-1), evect # 0 has greatest variance
#[n, bins, patches] = plot_pc_logarithmic(Q_reduced, evect_number, "test_name2") 

for i in range(20):
	plot_pc_logarithmic(Q_reduced, i, "Principal Component " + str(i+1))
