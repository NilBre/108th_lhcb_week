import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

# define dimensions
halfmodule_length = 4833 / 2  # in mm
halfmodule_width = 536  # in mm
num_modules_per_quarter = 5
n_quarter = 4
n_halves = 2
# local half module coordinates for top half modules
bottom_left_corner = [-halfmodule_width, 0, 0]
bottom_right_corner = [halfmodule_width, 0, 0]
top_left_corner = [-halfmodule_width, halfmodule_length, 0]
top_right_corner = [halfmodule_width, halfmodule_length, 0]

def make_full_layer_plot():
    fig, ax = plt.subplots()
    for module in range(num_modules_per_quarter): # top right
        ax.add_patch(Rectangle((bottom_left_corner[0] + ((module+1) * halfmodule_width), bottom_left_corner[1]), halfmodule_width, halfmodule_length, edgecolor='red', linewidth=2, fill=False))
        ax.set_xlim(-4000, 4000)
        ax.set_ylim(-3300, 3300)
    for module in range(num_modules_per_quarter): # top left
        ax.add_patch(Rectangle((bottom_left_corner[0] - (module * halfmodule_width), bottom_left_corner[1]), halfmodule_width, halfmodule_length, edgecolor='blue', linewidth=2, fill=False))
    for module in range(num_modules_per_quarter): # bottom right
        ax.add_patch(Rectangle((bottom_left_corner[0] + ((module+1) * halfmodule_width), bottom_left_corner[1] - halfmodule_length), halfmodule_width, halfmodule_length, edgecolor='green', linewidth=2, fill=False))
    for module in range(num_modules_per_quarter): # bottom left
        ax.add_patch(Rectangle((bottom_left_corner[0] - (module * halfmodule_width), bottom_left_corner[1] - halfmodule_length), halfmodule_width, halfmodule_length, edgecolor='black', linewidth=2, fill=False))
    plt.title('module fitting at the joint')
    plt.xlabel('global module position in x [mm]')
    plt.ylabel('global module position in y [mm]')
    plt.savefig('retest_uncertainty/out_rectangle/plot_global_positions.pdf')
    plt.clf()

def make_edges_plot(y_data_top, y_data_bottom):
    top_blc = [] # blc = bottom left corner for top half modules
    bottom_blc = [] # for bottomhalf modules
    for i in range(len(y_data_top)):
        if i < 5:
            top_blc.append([-halfmodule_width * (i+1), y_data_top[i], 0])
            bottom_blc.append([-halfmodule_width * (i+1), y_data_bottom[i], 0])
        else:
            top_blc.append([halfmodule_width * (i-5), y_data_top[i], 0])
            bottom_blc.append([halfmodule_width * (i-5), y_data_bottom[i], 0])
    fig, ax2 = plt.subplots()
    for module in range(len(top_blc)): # top
        ax2.add_patch(Rectangle((top_blc[module][0], top_blc[module][1]), halfmodule_width, 0, edgecolor='red', linewidth=1, fill=False))
        ax2.set_xlim(-2700, 2700)
        ax2.set_ylim(-1212, -1214)
    for module in range(len(bottom_blc)): # bottom
        ax2.add_patch(Rectangle((bottom_blc[module][0], bottom_blc[module][1]), halfmodule_width, 0, edgecolor='blue', linewidth=1, fill=False))
    plt.title('module fitting at the joint')
    plt.xlabel('global module position in x [mm]')
    plt.ylabel('global module position in y [mm]')
    plt.savefig('retest_uncertainty/out_rectangle/plot_edges_positions.pdf')
    plt.clf()

# example:
# hypothetical edges
y_final = [-1213.443, -1213.454, -1213.317, -1213.372, -1213.496, 
           -1213.431, -1213.472, -1213.502, -1213.402, -1213.374, 
           -1213.442, -1213.418, -1213.468, -1213.518, -1213.386, 
           -1213.391, -1213.404, -1213.499, -1213.431, -1213.390]

make_edges_plot(y_final[0:10], y_final[10:20])