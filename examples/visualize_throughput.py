'''
Loads saved on-axis and off-axis HWO core/total throughput, then (1) plots the pupil shapes and (2) overlays core vs total throughput curves vs source offset for multiple λ values.
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from matplotlib import rc
from matplotlib.colorbar import Colorbar 
import os
from pystarshade.config import OUTPUT_DIR, SCENES_DIR, DATA_DIR

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
def get_nearest(val, arr):
    return np.argmin(np.abs(arr - val))

fname_on = 'hwopupil_onaxis_259.npz'
fname_off = 'hwopupil_offaxis_259.npz'

file_path_on = os.path.join(DATA_DIR, 'pupils', fname_on)
file_path_off = os.path.join(DATA_DIR, 'pupils', fname_off)

data = np.load(file_path_on, allow_pickle=False)
pupil_onaxis = data['pupil']

data = np.load(file_path_off, allow_pickle=False)
pupil_offaxis = data['pupil']

fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# Plot the first image
axes[0].imshow(pupil_offaxis,extent=(-3, 3, -3, 3))
axes[0].set_xlabel('x [m]',fontsize=14)
axes[0].set_ylabel('y [m]',fontsize=14)
axes[0].set_title('HWO 6m pupil offaxis',fontsize=14)

# Plot the second image
axes[1].imshow(pupil_onaxis,extent=(-3, 3, -3, 3))
axes[1].set_xlabel('x [m]',fontsize=14)
axes[1].set_ylabel('y [m]',fontsize=14)
axes[1].set_title('HWO 6m pupil onaxis', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()

wl_ = [5e-7, 5.5e-7, 6e-7, 6.5e-7, 7e-7, 7.5e-7, 8e-7, 8.5e-7, 9e-7, 9.5e-7]
cmap = plt.get_cmap('Blues')
colors = [cmap(i / (len(wl_) + 4)) for i in range(len(wl_) + 4)]

cmap2 = plt.get_cmap('Greens')
colors2 = [cmap2(i / (len(wl_) + 4)) for i in range(len(wl_) + 4)]

through_fname_on = os.path.join(DATA_DIR, 'psf', 'hwo_throughput_hwopupil_onaxis_001m.npz')
through_fname_off = os.path.join(DATA_DIR, 'psf', 'hwo_throughput_hwopupil_offaxis_001m.npz')

data_throughput_on = np.load(through_fname_on, allow_pickle=False)
wl = data_throughput_on['wl']

N_wl, N, _ = np.shape(data_throughput_on['total_throughput'])
total_throughput_on = data_throughput_on['total_throughput'][:, N//2, N//2:]
core_throughput_on = data_throughput_on['core_throughput'][:, N//2, N//2:]

data_throughput_off = np.load(through_fname_off, allow_pickle=False)
total_throughput_off = data_throughput_off['total_throughput'][:, N//2, N//2:]
core_throughput_off = data_throughput_off['core_throughput'][:, N//2, N//2:]

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0)

ax1 = plt.subplot(gs[0, 0])
for i in range(len(wl_)):

    # First plot: core throughput
    ax1.plot(np.arange(len(core_throughput_off[i])) * 2, core_throughput_off[i], color=colors[i+4], linewidth=1., label='off-axis pupil')
    plt1 = ax1.plot(np.arange(len(core_throughput_on[i])) * 2, core_throughput_on[i], color=colors2[i+4], linewidth=1., label='on-axis pupil')

    max_val_on = np.max(core_throughput_on[i])
    loc_max_on = np.argmax(core_throughput_on[i])
    loc_max_off = get_nearest(max_val_on, core_throughput_off[i][:loc_max_on+1])

    if i == len(wl_)-1:
        plt.axvline(np.arange(len(core_throughput_on[-1]))[loc_max_on] * 2 , linestyle='dashed', color=colors2[i+4])
        plt.axvline(np.arange(len(core_throughput_on[-1]))[loc_max_off] * 2, linestyle='dashed', color=colors[i+4])

ax1.set_xlabel('Source [mas]', size=18)
ax1.set_ylabel('Core throughput', size=18)

cmappable = ScalarMappable(norm=Normalize(300,1000), cmap=cmap)
cmappable2 = ScalarMappable(norm=Normalize(300,1000), cmap=cmap2)
cbar_ax = fig.add_axes([0.02, 0.12, 0.01, .75])
cbar_ax2 = fig.add_axes([0.06, 0.12, 0.01, .75])
fig.colorbar( mappable = cmappable,  cax=cbar_ax, boundaries=[500, 600, 700, 800, 900, 1000])
fig.colorbar( mappable = cmappable2,  cax=cbar_ax2, boundaries=[500, 600, 700, 800, 900, 1000])
fig.text(0.005, 0.5, r'$\lambda$ [nm]', va='center', rotation='vertical', fontsize=14)
fig.text(0.055, 0.9, r'On-axis', va='center', rotation='horizontal', fontsize=14)
fig.text(0.005, 0.9, r'Off-axis', va='center', rotation='horizontal', fontsize=14)
fig.text(0.005, 0.93, r'Pupil:', va='center', rotation='horizontal', fontsize=14)

ax1 = plt.subplot(gs[0, 1])
for i in range(len(wl_)):
    # First plot: core throughput
    ax1.plot(np.arange(len(core_throughput_off[i])) * 2,total_throughput_off[i], color=colors[i+4], linewidth=1., label='off-axis pupil')
    ax1.plot(np.arange(len(core_throughput_on[i])) * 2, total_throughput_on[i], color=colors2[i+4], linewidth=1., label='on-axis pupil')
#    if i == 0: ax1.legend()

ax1.set_xlabel('Source [mas]', size=18)
ax1.set_ylabel(r'Total throughput', size=18)
plt.show()
