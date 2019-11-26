import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
rc('text', usetex=False)
import h5py
from qutip import*
plt.figure(figsize =[6,6])
#Enter directory and name of measurement

##############################################################################################################################
#Click on the points on screen to define an approximation line for interpolation
contrast_min = 0
contrast_max = 3
# #
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0224'
fname = 'Two_tone_85.hdf5'
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
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0224'
fname = 'Two_tone_86.hdf5'
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
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0225'
fname = 'Two_tone_86.hdf5'
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
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0225'
fname = 'Two_tone_103.hdf5'
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
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0225'
fname = 'Two_tone_104.hdf5'
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
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0225'
fname = 'Two_tone_106.hdf5'
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
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0225'
fname = 'Two_tone_87.hdf5'
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
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0226'
fname = 'One_tone_6.hdf5'
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
    demod_real = data[:,3,:]
    demod_imag = data[:,4,:]
    demod_mag = np.sqrt(demod_real**2 + demod_imag**2)
    demod_magdB = 20*np.log10(demod_mag)
    # demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    # demod_phase_norm = np.zeros(demod_phase.shape)
    # for idx in range(len(demod_phase[0,:])):
    #     demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))
    for idx in range(len(demod_mag[0,:])):
        demod_mag[:,idx] = abs((demod_mag[:,idx]-np.min(demod_mag[:,idx])))/(np.max(demod_mag[:,idx])-np.min(demod_mag[:,idx]))

Z = demod_mag
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = 0, vmax = 2)
###############################################################################################################
# Define constants
# e = 1.602e-19    #Fundamental charge
# h = 6.62e-34    #Placnk's constant
# phi_o = h/(2*e) #Flux quantum
#
# N = 50
# B_coeff = 60
# level_num = 10
# #Model
# def trans_energy(current, E_l, E_c, E_j, A, offset):
#     energies = np.zeros((len(current),level_num))
#     flux = current * B_coeff * A * 1e-4
#     phi_ext = (flux/phi_o-offset) * 2 * np.pi
#     a = tensor(destroy(N))
#     phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
#     na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
#     for idx in range(len(current)):
#         ope = 1.0j * (phi - phi_ext[idx])
#         H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
#         for idy in range(level_num):
#             energies[idx,idy] = H.eigenenergies()[idy]
#     return energies
#
# current_nice = np.linspace(0,1.2,601)*1e-3
# plt.figure(figsize = [5,5])
#Qubit A
# E_l=1.076188026577402
# E_c=1.0245922439188155
# E_j=4.817434033496316
# A=2.1415963616712136e-10
# offset=0.028043712988006286
# energies = trans_energy(current_nice, E_l, E_c, E_j, A, offset)
# plt.plot(current_nice*1e3, energies[:,1]-energies[:,0], '-', color = '#3399FF', label = 'Qubit A')
# plt.plot(current_nice*1e3, energies[:,2]-energies[:,0], '-', color = '#3399FF')
# plt.plot(current_nice*1e3, energies[:,3]-energies[:,0], '-', color = '#3399FF')
# plt.plot(current_nice*1e3, energies[:,3]-energies[:,0], '-', color = '#3399FF')
# plt.plot(current_nice*1e3, energies[:,4]-energies[:,0], '-', color = '#3399FF')
# plt.plot(current_nice*1e3, energies[:,5]-energies[:,0], '-', color = '#3399FF')
# plt.plot(current_nice*1e3, energies[:,2]-energies[:,1], '-', color = '#3399FF')
# #
# # # Qubit B
# E_l=1.4989850921941266
# E_c=1.0172194919268065
# E_j=5.257475161569078
# A=2.1086704960372626e-10
# offset=0.02347013794896157
# energies = trans_energy(current_nice, E_l, E_c, E_j, A, offset)
# plt.plot(current_nice*1e3, energies[:,1]-energies[:,0], '--', color = '#FF9933', label = 'Qubit B')
# plt.plot(current_nice*1e3, energies[:,2]-energies[:,0], '--', color = '#FF9933')
# plt.plot(current_nice*1e3, energies[:,3]-energies[:,0], '--', color = '#FF9933')
# plt.plot(current_nice*1e3, energies[:,4]-energies[:,0], '--', color = '#FF9933')
# plt.plot(current_nice*1e3, energies[:,5]-energies[:,0], '--', color = '#FF9933')
# plt.plot(current_nice*1e3, energies[:,2]-energies[:,1], '--', color = '#FF9933')
############################################################################

# e = 1.602e-19    #Fundamental charge
# h = 6.62e-34    #Placnk's constant
# phi_o = h/(2*e) #Flux quantum
#
# Na = 30
# Nb = 30
# B_coeff = 60
#
# E_la=1.0342744966054378
# E_ca=1.0489460953110028
# E_ja=4.849498118802835
# E_lb=1.5714757907680506
# E_cb=0.9853058152483324
# E_jb=5.322417547650805
# J_c=0.39114306373256086
#
# Aa=2.1954708195080567e-10
# offset_a=0.04251581069531609
#
# Ab=2.0511133265871804e-10
# offset_b=0.010276595744353665
#
level_num = 20
current = np.linspace(0,1.2,1201)*1e-3
# energies = np.zeros((len(current), level_num))
# #
# flux_a = current * B_coeff * Aa * 1e-4
# phi_ext_a = (flux_a/phi_o-offset_a) * 2 * np.pi
# flux_b = current * B_coeff * Ab * 1e-4
# phi_ext_b = (flux_b/phi_o-offset_b) * 2 * np.pi
#
# a = tensor(destroy(Na), qeye(Nb))
# phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
# na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)
#
# b = tensor(qeye(Na), destroy(Nb))
# phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
# nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)
#
directory = 'C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_AugustusVI_fit_20190226.txt"
path = directory + '\\' + fname
energies = np.genfromtxt(path)

# for idx in range(1,level_num):
#     plt.plot(current*1e3, energies[:,idx] - energies[:,0], color ='k', linestyle ='--', alpha = 0.6)

#Qubit transitions
# plt.plot(current*1e3, energies[:,1] - energies[:,0], color = 'k', linestyle ='--', label=r'$|00\rangle \rightarrow |10\rangle$')
# plt.plot(current*1e3, energies[:,2] - energies[:,0], color = 'k', linestyle ='--', label=r'$|00\rangle \rightarrow |01\rangle$')
#

#Transitions near gate
plt.plot(current*1e3, energies[:,1] - energies[:,0], color = 'k', linestyle ='-', label=r'$|00\rangle \rightarrow |10\rangle$')
plt.plot(current*1e3, energies[:,3] - energies[:,2], color = 'b', linestyle ='--', label=r'$|01\rangle \rightarrow |11\rangle$')
plt.plot(current*1e3, energies[:,2] - energies[:,0], color = 'k', linestyle ='-', label=r'$|00\rangle \rightarrow |20\rangle$')
plt.plot(current*1e3, energies[:,3] - energies[:,1], color = 'b', linestyle ='--', label=r'$|10\rangle \rightarrow |11\rangle$')
#
# plt.plot(current*1e3, energies[:,4] - energies[:,1], linestyle ='--', label=r'$|10\rangle \rightarrow |20\rangle$')
# plt.plot(current*1e3, energies[:,5] - energies[:,1], linestyle ='--', label=r'$|10\rangle \rightarrow |02\rangle$')
# plt.plot(current*1e3, energies[:,6] - energies[:,1], linestyle ='--', label=r'$|10\rangle \rightarrow |21\rangle$')
# plt.plot(current*1e3, energies[:,7] - energies[:,1], linestyle ='--', label=r'$|10\rangle \rightarrow |12\rangle$')
#
# plt.plot(current*1e3, energies[:,4] - energies[:,2], linestyle ='--', label=r'$|01\rangle \rightarrow |20\rangle$')
# plt.plot(current*1e3, energies[:,5] - energies[:,2], linestyle ='--', label=r'$|01\rangle \rightarrow |02\rangle$')
# plt.plot(current*1e3, energies[:,6] - energies[:,2], linestyle ='--', label=r'$|01\rangle \rightarrow |21\rangle$')
# plt.plot(current*1e3, energies[:,7] - energies[:,2], linestyle ='--', label=r'$|01\rangle \rightarrow |12\rangle$')
#
# plt.plot(current*1e3, energies[:,4] - energies[:,3], linestyle ='-', label=r'$|11\rangle \rightarrow |20\rangle$')
# plt.plot(current*1e3, energies[:,5] - energies[:,3], linestyle ='-', label=r'$|11\rangle \rightarrow |02\rangle$')
# plt.plot(current*1e3, energies[:,6] - energies[:,3], linestyle ='-', label=r'$|11\rangle \rightarrow |21\rangle$')
# plt.plot(current*1e3, energies[:,7] - energies[:,3],color = 'm', linestyle ='-',linewidth = 2, label=r'$|11\rangle \rightarrow |12\rangle$')


#transitions near cavity

# plt.plot(current*1e3, 7.5 - (energies[:,1] - energies[:,0]), color ='k', linestyle ='--', alpha = 1, label=r'$|00\rangle \rightarrow |10\rangle$' + ' RSB')
# plt.plot(current*1e3, 7.5 - (energies[:,2] - energies[:,0]), color ='k', linestyle ='-.', alpha = 1, label=r'$|00\rangle \rightarrow |01\rangle$'+ ' RSB')
#
# plt.plot(current*1e3, (energies[:,8] - energies[:,1]), linestyle ='--', alpha = 1, label = r'$|10\rangle \rightarrow |30\rangle$')
# plt.plot(current*1e3, (energies[:,9] - energies[:,1]), linestyle ='--', alpha = 1, label = r'$|10\rangle \rightarrow |03\rangle$')
# plt.plot(current*1e3, (energies[:,10] - energies[:,1]), linestyle ='--', alpha = 1, label = r'$|10\rangle \rightarrow |31\rangle$')
# plt.plot(current*1e3, (energies[:,11] - energies[:,1]), linestyle ='--', alpha = 1, label = r'$|10\rangle \rightarrow |13\rangle$')
# plt.plot(current*1e3, (energies[:,12] - energies[:,1]), linestyle ='--', alpha = 1, label = r'$|10\rangle \rightarrow |22\rangle$')
#
#
# plt.plot(current*1e3, (energies[:,9] - energies[:,2]), linestyle ='--', alpha = 1, label=r'$|01\rangle \rightarrow |03\rangle$')
# plt.plot(current*1e3, (energies[:,10] - energies[:,2]), linestyle ='--', alpha = 1, label=r'$|01\rangle \rightarrow |31\rangle$')
# plt.plot(current*1e3, (energies[:,11] - energies[:,2]), linestyle ='--', alpha = 1, label=r'$|01\rangle \rightarrow |13\rangle$')
# plt.plot(current*1e3, (energies[:,12] - energies[:,2]), linestyle ='--', alpha = 1, label=r'$|01\rangle \rightarrow |22\rangle$')
#
# plt.plot(current*1e3, (energies[:,10] - energies[:,3]), linestyle ='-', alpha = 1, label=r'$|11\rangle \rightarrow |31\rangle$')
# plt.plot(current*1e3, (energies[:,11] - energies[:,3]), linestyle ='-', alpha = 1, label=r'$|11\rangle \rightarrow |13\rangle$')
# plt.plot(current*1e3, (energies[:,12] - energies[:,3]), linestyle ='-', alpha = 1, label=r'$|11\rangle \rightarrow |22\rangle$')



plt.xticks(size=18.0)
# plt.xticks([0.75,0.85,0.95],size=18.0)
plt.yticks(size=18.0)
# plt.ylim([0,7])
# plt.xlim([0,1.2])
plt.legend(fontsize = 'medium')
plt.show()