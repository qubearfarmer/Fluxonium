import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
rc('text', usetex=False)
fig = plt.figure(figsize = [6,6])
import h5py
from qutip import*

########################################################################################
# directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0227'
# fname = 'Rabi_5.hdf5'
# path = directory + '\\' + fname
#
# #Read data and fit
# with h5py.File(path,'r') as hf:
#     # print('List of arrays in this file: \n', list(hf.keys()))
#     data_group = hf['Data']
#     # print (list(data_group.keys()))
#     channel_names = data_group['Channel names']
#     print (channel_names[0:])
#     data = data_group['Data']
#     time = data[:,0,0]
#     power = data[0,1,:]
#     demod_real = data[:,3,:]
#     demod_imag = data[:,4,:]
#     demod_mag = np.sqrt(demod_real**2 + demod_imag**2)
#     demod_magdB = 20*np.log10(demod_mag)
#     demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
#     demod_phase_norm = np.zeros(demod_phase.shape)
#     for idx in range(len(demod_phase[0,:])):
#         demod_phase_norm[:,idx] = ((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))
    # for idx in range(len(demod_phase[:,0])):
    #     demod_phase_norm[idx,:] = ((demod_phase[idx,:]-np.mean(demod_phase[idx,:])))
    # for idx in range(len(demod_mag[0,:])):
    #     demod_mag[:,idx] = abs((demod_mag[:,idx]-np.min(demod_mag[:,idx])))/(np.max(demod_mag[:,idx])-np.min(demod_mag[:,idx]))

# Z = demod_phase_norm.transpose()
# X,Y = np.meshgrid(time*1e9, power)
# plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = -1, vmax = 1)
########################################################################
# directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0228'
# fname = 'Rabi_5.hdf5'
# path = directory + '\\' + fname
#
# #Read data and fit
# with h5py.File(path,'r') as hf:
#     # print('List of arrays in this file: \n', list(hf.keys()))
#     data_group = hf['Data']
#     # print (list(data_group.keys()))
#     channel_names = data_group['Channel names']
#     print (channel_names[0:])
#     data = data_group['Data']
#     time = data[:,0,0]
#     power = data[0,1,:]
#     demod_real = data[:,3,:]
#     demod_imag = data[:,4,:]
#     demod_mag = np.sqrt(demod_real**2 + demod_imag**2)
#     demod_magdB = 20*np.log10(demod_mag)
#     demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
#     demod_phase_norm = np.zeros(demod_phase.shape)
#     for idx in range(len(demod_phase[0,:])):
#         demod_phase_norm[:,idx] = ((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))
#     # for idx in range(len(demod_phase[:,0])):
#     #     demod_phase_norm[idx,:] = ((demod_phase[idx,:]-np.mean(demod_phase[idx,:])))
#     # for idx in range(len(demod_mag[0,:])):
#     #     demod_mag[:,idx] = abs((demod_mag[:,idx]-np.min(demod_mag[:,idx])))/(np.max(demod_mag[:,idx])-np.min(demod_mag[:,idx]))
#
# Z = demod_phase_norm.transpose()
# X,Y = np.meshgrid(time*1e9, power+1)
# plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = -0.75, vmax = 0.75)

# plt.tick_params(labelsize = 16)
# plt.ylim([-28,-10])
# plt.yticks([-10,-15,-20,-25])
##########################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0226'
fname = 'Rabi_5.hdf5'
path = directory + '\\' + fname

# Read data and fit
with h5py.File(path,'r') as hf:
    # print('List of arrays in this file: \n', list(hf.keys()))
    data_group = hf['Data']
    # print (list(data_group.keys()))
    channel_names = data_group['Channel names']
    print (channel_names[0:])
    data = data_group['Data']
    time = data[:,0,0]
    demod_real = data[:,1,:]
    demod_imag = data[:,2,:]
    demod_mag = np.sqrt(demod_real**2 + demod_imag**2)
    demod_magdB = 20*np.log10(demod_mag)
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = ((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))
#     for idx in range(len(demod_phase[:,0])):
#         demod_phase_norm[idx,:] = ((demod_phase[idx,:]-np.mean(demod_phase[idx,:])))
#     for idx in range(len(demod_mag[0,:])):
#         demod_mag[:,idx] = abs((demod_mag[:,idx]-np.min(demod_mag[:,idx])))/(np.max(demod_mag[:,idx])-np.min(demod_mag[:,idx]))
cut=40
plt.plot(time[:cut]*1e9, demod_phase_norm[:cut,0],'-o')
plt.tick_params(labelsize = 18.0)
plt.yticks([])
plt.xlim([0,150])

####################
# def rabi_flop(t,freq,amp,tau, x1):
#     return amp*np.cos(2*np.pi*freq*(t - x1))*np.exp(-(t-x1)/tau)
# # print(demod_phase_norm.shape)
# from scipy.optimize import curve_fit
# # (time[np.argmax(demod_phase_norm)] - time[np.argmin(demod_phase_norm)])**-1
# data = demod_phase_norm[:cut,0]
# guess =([1.2e7, -0.5*(np.max(data) - np.min(data)) , 1e-6, 10e-9])
# opt, cov = curve_fit(rabi_flop, time[:cut], data, guess)
#
# time_nice = np.linspace(time[0], time[-1], 1001)
# plt.plot(time_nice*1e9, rabi_flop(time_nice, *opt),linestyle = '-' )

plt.show()