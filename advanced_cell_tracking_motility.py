# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:00:11 2024

@author: Nasser Ghazi 
"""
# %% import libraries


# for compatibility with Python 2 and 3
from __future__ import division, unicode_literals, print_function
import trackpy as tp
from scipy.optimize import curve_fit

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib qt
from pathlib import Path
import re

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Latin Modern Roman",
    "font.sans-serif": ["Helvetica"]})

# Optionally, tweak styles.
plt.rc('figure',  figsize=(8, 8))
plt.rc('image', cmap='gray')


# %% import data from imagej FindStackMaxima format


cell=r'' #cell line
cond=r''#'21\% O$_2$ ' # extra important condition. 

f = pd.read_csv(r"\Results.csv")
str_path = (r'')
path = Path(str_path)


# %% Define conditions constants linked to experiment from time-lapse microscopy

mag = 1.68 # In µm/pixel
dt = 60;  # In seconds

#beta_min max when filtering individual MSDs, if unsure set max=2 and min=0 to keep all trajectories
betamax=1.75;
betamin=0.55;
betaminn = str(betamin)
betamaxx = str(betamax)


# %% Assemble x and y each into their respective columns

"""
This step is to ensure the detected particles/cells are in correct format for usage with trackpy.
It is optimized for use with FindMaxima detection from ImageJ (FIJI)
"""

nbim = np.size(f, 0)

rows = []
for i in range(nbim):
    ncell = int(f.iloc[i, 1])
    for j in range(ncell):
        x = f.iloc[i, 2 * j + 2]
        y = f.iloc[i, 2 * j + 3]
        frame = f.iloc[i, 0]
        rows.append([x, y, frame])
    
    
    
    
# %% Turn back into dataframe//Sort//and Remove zeros

df = pd.DataFrame(rows, columns=['x', 'y', 'frame'])


#%% Keep only rows where 'frame' is less than or equal to x, if needed

# df = df[df['frame'] <= x]

#%% Get cell density based off of first frame 


"""
If needed, this section of the code can be used to directly compute the cell density,
it was written with assumption that the initial cell density is homogenous
"""


# # Define 1mm in terms of pixels
# length_in_um = 1000
# length_in_pixel = length_in_um / mag

# # Find the center of the frame
# max_x, max_y = df['x'].max(), df['y'].max()
# min_x, min_y = df['x'].min(), df['y'].min()

# center_x, center_y = (max_x + min_x) / 2, (max_y + min_y) / 2

# # Define center square boundaries
# x_lower = center_x - length_in_pixel / 2
# x_upper = center_x + length_in_pixel / 2
# y_lower = center_y - length_in_pixel / 2
# y_upper = center_y + length_in_pixel / 2

# # Filter center square and count cells
# df_filtered = df[(df['x'] >= x_lower) & (df['x'] <= x_upper) & (df['y'] >= y_lower) & (df['y'] <= y_upper)]
# count_first_frame = len(df_filtered[df_filtered['frame'] == df['frame'].max()])

# # Initialize list to store cell counts and rectangle names
# cell_counts = [count_first_frame]
# rect_names = ['Rectangle1']

# # Randomly sample 5 additional squares
# for i in range(1, 6):
#     random_x_lower = np.random.uniform(min_x + length_in_pixel / 2, max_x - length_in_pixel / 2)
#     random_y_lower = np.random.uniform(min_y + length_in_pixel / 2, max_y - length_in_pixel / 2)

#     # Define square boundaries
#     random_x_upper = random_x_lower + length_in_pixel
#     random_y_upper = random_y_lower + length_in_pixel

#     # Filter and count cells
#     random_count = len(df[(df['x'] >= random_x_lower) & (df['x'] <= random_x_upper) &
#                           (df['y'] >= random_y_lower) & (df['y'] <= random_y_upper) &
#                           (df['frame'] == df['frame'].max())])
    
#     # Append to cell_counts and rect_names
#     cell_counts.append(random_count)
#     rect_names.append(f"Rectangle{i+1}")

# # Calculate average and error
# density = np.mean(cell_counts)
# error = np.std(cell_counts, ddof=1)

# # Print and save results
# print(f"Average Density: {density} cells/mm^2")
# print(f"Standard Error: {error}")

# # Ensure the directory exists
# path.mkdir(parents=True, exist_ok=True)

# # Save to Log.txt in the current working path
# with open(path / 'Log.txt', 'w') as f:
#     f.write("Rectangle,MaximaCount\n")
#     for name, count in zip(rect_names, cell_counts):
#         f.write(f"{name},{count}\n")
#     f.write(f"\nAverageDensity,{density}\n")
#     f.write(f"StandardError,{error}\n")



#%% link trajectories

"""
usage of trackpy to link trajectories
parameters should be fine-tuned according to the sample cells being observed
for more information: https://github.com/soft-matter/trackpy
"""

max_dist_allowed = 15
t = tp.link(df, search_range=max_dist_allowed, memory=3, pos_columns=['x', 'y'], 
            t_column='frame',adaptive_stop=0.1,adaptive_step=0.99)


# %% Get number of frames

slices = df.iloc[0:-1, 2]
s = slices.tolist()
frames = pd.unique(s)
framenb = len(frames)
print(framenb)

"""
When working with MSD extraction, it is useful to take 1/3 or 1/2 of your total movie
as the "good" parameter. This is to ensure good statistics as the lag time increases
in the later stages of iteration

"""

good = round(framenb/3) 


#%% Filter trajectories that are succesively tracked in the linking stage to be at least as long as "good" param.

traj_minframe_req = good
t1 = tp.filter_stubs(t, traj_minframe_req)

# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

#a preference
td = t1.copy()

#%% Plot unfiltered trajectories to compare later

fig, ax = plt.subplots()
ax.set(title='Unfiltered Trajectories')
tp.plot_traj(t1);
fig.savefig(f"{path}/unfiltered_trajectories.png", dpi=150)


fig, ax = plt.subplots()
ax.set(title='Stub Filtered Trajectories')
tp.plot_traj(td);
fig.savefig(f"{path}/stub_filtered_trajectories.png", dpi=150)

#%% A unique custom-filter, to keep trajectories that are not immobile.

"""

This section was made to help researchers/scientists when faced with experimental difficulties:
    i.e. dead cells, dust, cyst
It ensures immobile trajectories are filtered out to not disturb the parameter extraction 
and is to be used only if needed

It computes the total displacement and simply filters out trajectories that have
moved less than "x" threshold

For simplicity, can keep threshold_distance = 0 if filtering isn't needed

"""

# Calculate the total displacement for each particle
def total_displacement(group):
    x_start, y_start = group.iloc[0][['x', 'y']]
    x_end, y_end = group.iloc[-1][['x', 'y']]
    return np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)

total_displacements = td.groupby('particle').apply(total_displacement)

# Filter out particles with a total displacement less than a threshold
threshold_distance = 0  # Adjust this based on your specific needs
filtered_particles = total_displacements.index[total_displacements >= threshold_distance]

# Create a filtered DataFrame
td_filtered = td[td['particle'].isin(filtered_particles)]
print('After total displacement filter:', td_filtered['particle'].nunique())  # Debug


# Show the plot
plt.show()
fig1, ax1 = plt.subplots()
ax1.set(title='Displacement Filtered Trajectories')
# Plot the trajectories using the color parameter
tp.plot_traj(td_filtered, pos_columns=['x','y'])
ax1.set_aspect('equal')
fig1.savefig(f"{path}/displacement_filtered_trajectories.png", dpi=150)

plt.close()



# %% Compute drift

"""
This is trackpy's built-in drift filter

"""

# d = tp.compute_drift(t2,1)
# d.plot()
# plt.show()

# %% TM = T2

#A preference of mine, to keep copies of dataframes for error-correction at future steps

tm = td_filtered.copy()


# %% Obtain full msd spectrum before filtering with beta values

"""
The extraction of individual MSDs for each cell/particle in the dataframe
Notice, there is no direct usage of the ensemble MSD arising from trackpy, 
this is due to its lack of beta filtering

We first take the individual MSDs:
    
"""


lagtime_max = good #nbim/3
# MSD of each individual cell/particle
im = tp.imsd(tm, mag, 1/dt, max_lagtime=int(lagtime_max))  #maxlagtime is in frames

# %% Plot full MSD Spectrum

"""
The full individual MSD of each cell plotted
"""

# fig, ax = plt.subplots()
# ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
# ax.set(title=' Unfiltered Individual MSDs ' + cell + cond, ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
#        xlabel='lag time $t$')
# matplotlib.pyplot.xticks(fontsize=14)
# matplotlib.pyplot.yticks(fontsize=14)
# ax.set_xlabel(r'Lag Time $t$ [sec]',fontsize = 15)
# ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',fontsize = 15)
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig.savefig(f"{path}/full_msd_unfiltered_raw.png", dpi=150)




# %% Function to filter using beta_min beta_max

"""
Here is a custom MSD filter I wrote to filter based on beta_min/max
It is used to update the trajectory dataframe as well
This is a preference for final cell trajectory plotting representative 
    of the individual and total ensemble MSD
""" 

from scipy.stats import linregress

def filter_msd_and_trajectories_by_beta(i_msd, tmi, betamin, betamax):
    filtered_msd = pd.DataFrame()
    filtered_trajectories = pd.DataFrame()
    
    for i in i_msd:
        one_third_length = int(len(i_msd[i]))
        
        y = np.log(i_msd[i].iloc[:one_third_length].dropna())
        x = np.log(i_msd[i].iloc[:one_third_length].dropna().index)
        
        slope, _, _, _, _ = linregress(x, y)
        
        if betamin <= slope <= betamax:
            filtered_msd[i] = i_msd[i]
            filtered_trajectories = filtered_trajectories.append(tmi[tmi.particle == i])
    
    return filtered_msd, filtered_trajectories



#%%

# Apply the MSD filtering function
betamin_value = 0.55  # Set minimum beta value
betamax_value = 1.75  # Set maximum beta value
imi, tmi = filter_msd_and_trajectories_by_beta(im, tm, betamin_value, betamax_value)

"""
imi is the new individual MSD dataset
tmi is the new corresponding trajectories
"""

#get the number of trajectories filtered as a percentage of the total initial number 
filtpc = str(round((1-np.size(imi)/np.size(im))*100,2)); 
print(filtpc)


#%% Advanced trajectory plotting


"""
My preferred way to share cell/particle trajectory plots

Each cell is color coded and mapped to the total time of the experiment by 
time normalization

For example, trajectory segments in blue reflect that initial time and in those
in red reflect the final time (ie t=0 and t=6h)

"""


from matplotlib.collections import LineCollection

# Normalize the time column based on the real min and max frame of all cells
tmi['norm_time'] = (tmi['frame'] - tmi['frame'].min()) / (tmi['frame'].max() - tmi['frame'].min())

# Convert x and y to µm
tmi['x_um'] = tmi['x']*mag
tmi['y_um'] = tmi['y']*mag

# Create the figure and axis
fig, ax = plt.subplots()

# Set the aspect ratio to equal
ax.set_aspect("equal")

# Create a color map
color_map = plt.get_cmap("turbo")

# Plot each trajectory using LineCollection
for name, group in tmi.groupby("particle"):
    points = np.array([group["x_um"], group["y_um"]]).T.reshape(-1, 1, 2)  # transpose to match format required by LineCollection
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=color_map, norm=plt.Normalize(tmi["norm_time"].min(), tmi["norm_time"].max()))
    lc.set_array(group["norm_time"])
    ax.add_collection(lc)

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(tmi["norm_time"].min(), tmi["norm_time"].max()))
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.05, shrink=0.5)  # Adjust the pad and shrink parameters to fit the size of the colorbar
cbar.set_label("Time")


# Add x and y labels in microns
ax.set_xlabel("X Position (µm)")
ax.set_ylabel("Y Position (µm)")

# Adjust the size of the plot
plt.subplots_adjust(right=0.8)

plt.xlim(tmi['x_um'].min(), tmi['x_um'].max())
plt.ylim(tmi['y_um'].max(), tmi['y_um'].min())

plt.title("Visualization of Particle Trajectories with Time-normalized Color Mapping", size=14)
plt.tight_layout()
plt.show()

fig.savefig(f"{path}/time_mapped_filtered_trajectories_final.png", dpi=150)



#%% Compute ensemble MSD

"""
The correct simple method to compute the ensemble MSD, the mean over
each lag time of all individual valid MSDs
"""

em = imi.mean(axis=1)

#%% Instantaneous velocity

"""
One approach to obtaining the inst. velocity of particles/cells
is to simply compute the sqrt(MSD)/(dt=first time step)
"""

vinst = str(round(np.sqrt(em.iloc[0])/(dt/60),2));
print(vinst)


#%% SAVE DATA

"""
Save data and refer to my other code, for reading this computed data for 
visualization, Diffusion coefficient and persistence time extraction, VACF, etc

"""

# Save im (unfiltered individual MSDs) to a CSV file in str_path
im.to_csv(f'{str(path)}/2unfiltered_individual_msds.csv', index=True)

# Save imi (filtered and kept MSDs) to a CSV file in str_path
imi.to_csv(f'{str(path)}/2filtered_kept_msds.csv', index=True)

# Save em (experimental average MSD) to a CSV file in str_path
em.to_csv(f'{str(path)}/2experimental_avg_msd.csv', index=True)

# Save tmi (filtered trajectories) to a CSV file in str_path
tmi.to_csv(f'{str(path)}/2filtered_trajectories.csv', index=False)
