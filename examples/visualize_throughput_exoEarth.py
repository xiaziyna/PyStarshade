import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from matplotlib import rc
from matplotlib.colorbar import Colorbar 
import os

#visualize the throughput at two single wavelengths with an exo-Earth overlaid

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '..', 'pystarshade', 'data'))

through_fname_on = os.path.join(data_dir, 'psf', 'hwo_throughput_hwopupil_onaxis_001m.npz')
through_fname_off = os.path.join(data_dir, 'psf', 'hwo_throughput_hwopupil_offaxis_001m.npz')

data_throughput_on = np.load(through_fname_on)
wl = data_throughput_on['wl']
print (wl)
d_mas = data_throughput_on['d_pix_mas']

N_wl, N, _ = np.shape(data_throughput_on['total_throughput'])
total_throughput_on = data_throughput_on['total_throughput'][:, N//2, N//2:]
core_throughput_on = data_throughput_on['core_throughput'][:, N//2, N//2:]

data_throughput_off = np.load(through_fname_off)
total_throughput_off = data_throughput_off['total_throughput'][:, N//2, N//2:]
core_throughput_off = data_throughput_off['core_throughput'][:, N//2, N//2:]

mas_to_rad = 4.84814e-9
wl_ = [5e-7, 5.5e-7, 6e-7, 6.5e-7, 7e-7, 7.5e-7, 8e-7, 8.5e-7, 9e-7, 9.5e-7]

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0)

ax1 = plt.subplot(gs[0, 0])
ax1.text(0.1, .76, '500 nm', fontsize = 14)
wl_i = 0
ax1.plot(np.arange(len(core_throughput_off[-1])) * d_mas / ((500e-9 / 6)/mas_to_rad), core_throughput_off[wl_i], linestyle='-',color='#3d8dc4', linewidth=2., label='off-axis pupil')
ax1.plot(np.arange(len(core_throughput_on[-1])) * d_mas / ((500e-9 / 6)/mas_to_rad), core_throughput_on[wl_i], linestyle='-', color='#7d77b7', linewidth=2., label='on-axis pupil')
ax1.axvline( 83/((500e-9 / 6)/mas_to_rad), label ='exoEarth (1AU sep. at 12 pc)', linestyle='dashed', color = 'green')
ax1.set_ylim((0, 0.8))
ax1.set_xlim((0, 7))


#ax1.set_xlabel('source [mas]', size=18)
ax1.set_xlabel(r'angular separation ($\lambda / D$)', size=18)
ax1.set_ylabel('core throughput', size=18)

ax1 = plt.subplot(gs[0, 1])
ax1.text(0.05, .76, '950 nm', fontsize=14)
wl_i = len(wl_) - 1
ax1.plot(np.arange(len(core_throughput_off[-1])) * d_mas / ((950e-9 / 6)/mas_to_rad), core_throughput_off[wl_i], linestyle='-',color='#3d8dc4', linewidth=2., label='off-axis pupil')
ax1.plot(np.arange(len(core_throughput_on[-1])) * d_mas / ((950e-9 / 6)/mas_to_rad), core_throughput_on[wl_i], linestyle='-', color='#7d77b7', linewidth=2., label='on-axis pupil')
ax1.axvline(83/((950e-9 / 6)/mas_to_rad), label ='exoEarth',linestyle='dashed', color = 'green')
ax1.set_xticks([0,1,2,3,4], [0,1,2,3,4])  # Example x-ticks
ax1.set_ylim((0, 0.8))
ax1.set_xlim((0, 4))

ax1.set_xlabel(r'angular separation ($\lambda / D$)', size=18)
ax1.set_ylabel('core throughput', size=18)

plt.legend(fontsize=12)
plt.show()
