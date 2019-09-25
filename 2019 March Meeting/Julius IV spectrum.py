import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from scipy.optimize import curve_fit
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################
contrast_min = -3
contrast_max = 3
flux_shift = 0.055
# plt.figure(figsize =[7,5])
directory = 'G:\Projects\Fluxonium\Data\\2019\\06\Data_0629\Two_tone_twpa_on_40.hdf5'

path = directory

#Read data and fit
with h5py.File(path,'r') as hf:
    # print('List of arrays in this file: \n', list(hf.keys()))
    data_group = hf['Data']
    # print (list(data_group.keys()))
    channel_names = data_group['Channel names']
    # print (channel_names[0:])
    data = data_group['Data']
    freq = data[:,0,0]
    current = data[0,1,:]
    demod_real = data[:,2,:]
    demod_imag = data[:,3,:]
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3-flux_shift,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
directory = 'G:\Projects\Fluxonium\Data\\2019\\06\Data_0630\Two_tone_twpa_on_40.hdf5'

path = directory

#Read data and fit
with h5py.File(path,'r') as hf:
    # print('List of arrays in this file: \n', list(hf.keys()))
    data_group = hf['Data']
    # print (list(data_group.keys()))
    channel_names = data_group['Channel names']
    # print (channel_names[0:])
    data = data_group['Data']
    freq = data[:,0,0]
    current = data[0,1,:]
    demod_real = data[:,2,:]
    demod_imag = data[:,3,:]
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3-flux_shift,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
directory = 'G:\Projects\Fluxonium\Data\\2019\\06\Data_0630\Two_tone_twpa_on_41.hdf5'

path = directory

#Read data and fit
with h5py.File(path,'r') as hf:
    # print('List of arrays in this file: \n', list(hf.keys()))
    data_group = hf['Data']
    # print (list(data_group.keys()))
    channel_names = data_group['Channel names']
    # print (channel_names[0:])
    data = data_group['Data']
    freq = data[:,0,0]
    current = data[0,1,:]
    demod_real = data[:,2,:]
    demod_imag = data[:,3,:]
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3-flux_shift,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\\2019\\07\Data_0703\Two_tone_twpa_on_43.hdf5'

path = directory

#Read data and fit
with h5py.File(path,'r') as hf:
    # print('List of arrays in this file: \n', list(hf.keys()))
    data_group = hf['Data']
    # print (list(data_group.keys()))
    channel_names = data_group['Channel names']
    # print (channel_names[0:])
    data = data_group['Data']
    freq = data[:,0,0]
    current = data[0,1,:]
    demod_real = data[:,2,:]
    demod_imag = data[:,3,:]
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)

###########################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 30
B_coeff = 50
E_l = 0.5825088902476563
E_c = 1.0019222206424947
E_j = 3.4376199557356957
A = 1.8682912465136252e-10  # in m^2
offset = -0.0027749398327730473

current = np.linspace(-0.05,1.4,101)*1e-3
energy = np.zeros((len(current),10))

flux = current * B_coeff * A * 1e-4
phi_ext = (flux/phi_o-offset) * 2 * np.pi
a = tensor(destroy(N))
phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
for idx in range(len(current)):
    ope = 1.0j * (phi - phi_ext[idx])
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
    for idy in range(10):
        energy[idx,idy] = H.eigenenergies()[idy]
cut = 60
plt.plot(current*1e3, energy[:,1] - energy[:,0],'k--', alpha = 0.8)
plt.plot(current*1e3, energy[:,2] - energy[:,0],'k--', alpha = 0.8)
plt.plot(current*1e3, energy[:,3] - energy[:,0],'k--', alpha = 0.8)
plt.plot(current*1e3, energy[:,4] - energy[:,0],'k--', alpha = 0.8)
plt.plot(current[cut:]*1e3, energy[cut:,2] - energy[cut:,1],'b--', alpha = 0.8)
plt.plot(current[cut:]*1e3, energy[cut:,3] - energy[cut:,1],'b--', alpha = 0.8)
###########################################################################
# plt.xlabel('Current (mA)', fontsize = 18.0)
# plt.ylabel('Frequency (GHz)', fontsize = 18.0)
plt.xlim([0,1.4])
plt.ylim([0,6])
plt.xticks([0,0.4, 0.8, 1.2])
plt.yticks([0,2,4,6])
plt.tick_params(labelsize = 16.0)
plt.show()