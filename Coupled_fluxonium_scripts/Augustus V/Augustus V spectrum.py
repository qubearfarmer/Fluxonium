from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import h5py

##########################################################################################
contrast_min = -2
contrast_max = 2
plt.figure(figsize=[10, 6])

directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_10.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_24.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_25.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_26.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_27.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_28.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_29.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_30.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0108'
fname = 'Two_tone_31.hdf5'
path = directory + '\\' + fname

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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
# directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0109'
# fname = 'Two_tone_33.hdf5'
# path = directory + '\\' + fname
#
# #Read data and fit
# with h5py.File(path,'r') as hf:
#     # print('List of arrays in this file: \n', list(hf.keys()))
#     data_group = hf['Data']
#     # print (list(data_group.keys()))
#     channel_names = data_group['Channel names']
#     # print (channel_names[0:])
#     data = data_group['Data']
#     freq = data[:,0,0]
#     current = data[0,1,:]
#     demod_real = data[:,2,:]
#     demod_imag = data[:,3,:]
#     demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
#     demod_phase_norm = np.zeros(demod_phase.shape)
#     for idx in range(len(demod_phase[0,:])):
#         demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])
#
# Z = demod_phase_norm
# X,Y = np.meshgrid(current*1e3,freq*1e-9)
# plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################
# directory = 'G:\Projects\Fluxonium\Data\Augustus V\\2019\\01\Data_0109'
# fname = 'Two_tone_35.hdf5'
# path = directory + '\\' + fname
#
# #Read data and fit
# with h5py.File(path,'r') as hf:
#     # print('List of arrays in this file: \n', list(hf.keys()))
#     data_group = hf['Data']
#     # print (list(data_group.keys()))
#     channel_names = data_group['Channel names']
#     # print (channel_names[0:])
#     data = data_group['Data']
#     freq = data[:,0,0]
#     current = data[0,1,:]
#     demod_real = data[:,2,:]
#     demod_imag = data[:,3,:]
#     demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
#     demod_phase_norm = np.zeros(demod_phase.shape)
#     for idx in range(len(demod_phase[0,:])):
#         demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])
#
# Z = demod_phase_norm
# X,Y = np.meshgrid(current*1e3,freq*1e-9)
# plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)

####################################################################################

# Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 30
B_coeff = 30
level_num = 20
#Model
def trans_energy(current, E_l, E_c, E_j, A, offset):
    energies = np.zeros((len(current),level_num))
    flux = current * B_coeff * A * 1e-4
    phi_ext = (flux/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current)):
        ope = 1.0j * (phi - phi_ext[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        for idy in range(level_num):
            energies[idx,idy] = H.eigenenergies()[idy]
    return energies

# current_nice = np.linspace(0,1.8,601)*1e-3
# Qubit A
# E_l=0.6066902029020371
# E_c=0.9933251002064469
# E_j=3.8871801583388383
# A=2.897481018144399e-10
# offset=0.04092744050650806
# energies = trans_energy(current_nice, E_l, E_c, E_j, A, offset)
# plt.plot(current_nice*1e3, energies[:,1]-energies[:,0], '--', color = '#3399FF', label = 'Qubit A')
# plt.plot(current_nice*1e3, energies[:,2]-energies[:,0], '--', color = '#3399FF')
# plt.plot(current_nice*1e3, energies[:,3]-energies[:,0], '--', color = '#3399FF')
# plt.plot(current_nice[-250:]*1e3, energies[-250:,2]-energies[-250:,1], '--', color = '#3399FF')

# Qubit B
# E_l=0.7072173289595397
# E_c=0.9963623769095454
# E_j=4.031775449444628
# A=2.3780519915479326e-10
# offset=0.030411108243708962
# energies = trans_energy(current_nice, E_l, E_c, E_j, A, offset)
# plt.plot(current_nice*1e3, energies[:,1]-energies[:,0], '--', color = '#FF9933', label = 'Qubit B')
# plt.plot(current_nice*1e3, energies[:,2]-energies[:,0], '--', color = '#FF9933')
# plt.plot(current_nice*1e3, energies[:,3]-energies[:,0], '--', color = '#FF9933')
# plt.plot(current_nice[-250:]*1e3, energies[-250:,2]-energies[-250:,1], '--', color = '#FF9933')
# plt.xlim([0,1.8])
# plt.ylim([0,6.5])
# plt.yticks([0,2,4,6])
# plt.xticks(np.linspace(0,1.8,4))

directory = 'C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_AugustusV_Jc=273.txt"
path = directory + '\\' + fname
energies = np.genfromtxt(path)
current = np.linspace(0, 2, 2000)*1e-3

for idx in range(1,level_num):
    plt.plot(current*1e3, (energies[:,idx] - energies[:,0]), color='k', linestyle ='--', alpha = 1)
plt.plot(current[1400:1800]*1e3, (energies[1400:1800,4] - energies[1400:1800,1]),linestyle ='--', alpha = 1)
plt.plot(current[1400:1800]*1e3, (energies[1400:1800,9] - energies[1400:1800,1]),linestyle ='--', alpha = 1)
plt.plot(current[1400:1800]*1e3, (energies[1400:1800,13] - energies[1400:1800,1]),linestyle ='--', alpha = 1)
# plt.plot(current[1000:1800]*1e3, (energies[1000:1800,5] - energies[1000:1800,2]),linestyle ='--', alpha = 1)
# plt.plot(current[1000:1800]*1e3, (energies[1000:1800,7] - energies[1000:1800,2]),linestyle ='--', alpha = 1)
# for idx in range(2,level_num):
#     plt.plot(current*1e3, (energies[:,idx] - energies[:,1]), linestyle ='--', alpha = 1, label = str(idx))

plt.legend()
plt.tick_params(labelsize = 18.0)
plt.show()
