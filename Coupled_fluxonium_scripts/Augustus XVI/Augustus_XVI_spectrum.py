import numpy as np
import matplotlib.pyplot as plt
import h5py
from qutip import*
import matplotlib.cm as cm
from matplotlib import rc
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
rc('text', usetex=False)
############################################################################################
#####################################################################
contrast_max = 8
contrast_min = -8
#File path
path = "G:\Projects\Fluxonium\Data\\2019\\08\Data_0830\Two_tone_23.hdf5"
f = Labber.LogFile(path)
freq = f.getData('Qubit RF - Frequency')[0]/1e9
current = f.getData('YOKO 1 - Current')[:,0]*1e3
signal = f.getData('Signal Demodulation - Value')
signal_phase = np.angle(signal)
for idx in range(len(current)):
    signal_phase[idx,:] = signal_phase[idx,:] - np.mean(signal_phase[idx,:])
for idx in range(len(freq)):
    signal_phase[:, idx] = signal_phase[:, idx] - np.mean(signal_phase[:, idx])
signal_phase = signal_phase*180/np.pi
X,Y = np.meshgrid(current, freq)
Z = signal_phase.transpose()
plt.pcolormesh(X,Y,Z, cmap = 'RdBu', vmax = contrast_max, vmin = contrast_min)
######################################################################
#File path
contrast_max = 5
contrast_min = -5
path = "G:\Projects\Fluxonium\Data\\2019\\08\Data_0830\Two_tone_24.hdf5"
f = Labber.LogFile(path)
freq = f.getData('Qubit RF - Frequency')[0]/1e9
current = f.getData('YOKO 1 - Current')[:,0]*1e3
signal = f.getData('Signal Demodulation - Value')
signal_phase = np.angle(signal)
for idx in range(len(current)):
    signal_phase[idx,:] = signal_phase[idx,:] - np.mean(signal_phase[idx,:])
# for idx in range(len(freq)):
#     signal_phase[:, idx] = signal_phase[:, idx] - np.mean(signal_phase[:, idx])
signal_phase = signal_phase*180/np.pi
X,Y = np.meshgrid(current, freq)
Z = signal_phase.transpose()
plt.pcolormesh(X,Y,Z, cmap = 'RdBu', vmax = contrast_max, vmin = contrast_min)
# plt.colorbar()
######################################################################
contrast_max = 5
contrast_min = -5
path = "G:\Projects\Fluxonium\Data\\2019\\08\Data_0830\Two_tone_25.hdf5"
f = Labber.LogFile(path)
freq = f.getData('Qubit RF - Frequency')[0]/1e9
current = f.getData('YOKO 1 - Current')[:,0]*1e3
signal = f.getData('Signal Demodulation - Value')
signal_phase = np.angle(signal)
for idx in range(len(current)):
    signal_phase[idx,:] = signal_phase[idx,:] - np.mean(signal_phase[idx,:])
for idx in range(len(freq)):
    signal_phase[:, idx] = signal_phase[:, idx] - np.mean(signal_phase[:, idx])
signal_phase = signal_phase*180/np.pi
X,Y = np.meshgrid(current, freq)
Z = signal_phase.transpose()
plt.pcolormesh(X,Y,Z, cmap = 'RdBu', vmax = contrast_max, vmin = contrast_min)
# plt.colorbar()
######################################################################
contrast_max = 5
contrast_min = -5
path = "G:\Projects\Fluxonium\Data\\2019\\09\Data_0901\Two_tone_31.hdf5"
f = Labber.LogFile(path)
freq = f.getData('Qubit RF - Frequency')[0]/1e9
current = f.getData('YOKO 1 - Current')[:,0]*1e3
signal = f.getData('Signal Demodulation - Value')
signal_phase = np.angle(signal)
for idx in range(len(current)):
    signal_phase[idx,:] = signal_phase[idx,:] - np.mean(signal_phase[idx,:])
# for idx in range(len(freq)):
#     signal_phase[:, idx] = signal_phase[:, idx] - np.mean(signal_phase[:, idx])
signal_phase = signal_phase*180/np.pi
X,Y = np.meshgrid(current, freq)
Z = signal_phase.transpose()
plt.pcolormesh(X,Y,Z, cmap = 'RdBu', vmax = contrast_max, vmin = contrast_min)
# plt.colorbar()
######################################################################
contrast_max = 20
contrast_min = -30
path = "G:\Projects\Fluxonium\Data\\2019\\09\Data_0905\Two_tone_89.hdf5"
f = Labber.LogFile(path)
freq = f.getData('Qubit RF - Frequency')[0]/1e9
current = f.getData('YOKO 1 - Current')[:,0]*1e3
signal = f.getData('Signal Demodulation - Value')
signal_phase = np.angle(signal)
for idx in range(len(current)):
    signal_phase[idx,:] = signal_phase[idx,:] - np.mean(signal_phase[idx,:])
# for idx in range(len(freq)):
#     signal_phase[:, idx] = signal_phase[:, idx] - np.mean(signal_phase[:, idx])
signal_phase = signal_phase*180/np.pi
X,Y = np.meshgrid(current, freq)
Z = signal_phase.transpose()
plt.pcolormesh(X,Y,Z, cmap = 'RdBu', vmax = contrast_max, vmin = contrast_min)
######################################################################
contrast_max = 10
contrast_min = -10
path = "G:\Projects\Fluxonium\Data\\2019\\09\Data_0905\Two_tone_90.hdf5"
f = Labber.LogFile(path)
freq = f.getData('Qubit RF - Frequency')[0]/1e9
current = f.getData('YOKO 1 - Current')[:,0]*1e3
signal = f.getData('Signal Demodulation - Value')
signal_phase = np.angle(signal)
for idx in range(len(current)):
    signal_phase[idx,:] = signal_phase[idx,:] - np.mean(signal_phase[idx,:])
# for idx in range(len(freq)):
#     signal_phase[:, idx] = signal_phase[:, idx] - np.mean(signal_phase[:, idx])
signal_phase = signal_phase*180/np.pi
X,Y = np.meshgrid(current, freq)
Z = signal_phase.transpose()
plt.pcolormesh(X,Y,Z, cmap = 'RdBu', vmax = contrast_max, vmin = contrast_min)
######################################################################
contrast_max = 5
contrast_min = -5
path = "G:\Projects\Fluxonium\Data\\2019\\09\Data_0906\Two_tone_90.hdf5"
f = Labber.LogFile(path)
freq = f.getData('Qubit RF - Frequency')[0]/1e9
current = f.getData('YOKO 1 - Current')[:,0]*1e3
signal = f.getData('Signal Demodulation - Value')
signal_phase = np.angle(signal)
for idx in range(len(current)):
    signal_phase[idx,:] = signal_phase[idx,:] - np.mean(signal_phase[idx,:])
# for idx in range(len(freq)):
#     signal_phase[:, idx] = signal_phase[:, idx] - np.mean(signal_phase[:, idx])
signal_phase = signal_phase*180/np.pi
X,Y = np.meshgrid(current, freq)
Z = signal_phase.transpose()
plt.pcolormesh(X,Y,Z, cmap = 'RdBu', vmax = contrast_max, vmin = contrast_min)
#####################################################################################
directory = 'C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_AugustusXVI_fit_20190906.txt"
path = directory + '\\' + fname

#####################################################################################
# e = 1.602e-19    #Fundamental charge
# h = 6.62e-34    #Placnk's constant
# phi_o = h/(2*e) #Flux quantum
#
# Na = 20
# Nb = 20
# B_coeff = 45
#
# E_la=1.1005421433121607
# E_ca=1.0642461566822103
# E_ja=4.560531050428476
#
# E_lb=1.8715619169771045
# E_cb=1.043549524435553
# E_jb=5.057933424197382
#
# Aa=2.611486727475769e-10
# offset_a=0.3286938204192708
#
# Ab=2.510898275538273e-10
# offset_b=0.2987656490418759
#
# J_c = 0.5
# level_num = 20
# current = np.linspace(0,2,100)*1e-3
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


#####################################################################################
# for idx in range(len(current)):
#     ope_a = 1.0j * (phi_a - phi_ext_a[idx])
#     Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
#     ope_b = 1.0j * (phi_b - phi_ext_b[idx])
#     Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
#     Hc = J_c * na * nb
#     H = Ha + Hb + Hc
#     for idy in range(level_num):
#         energies[idx,idy] = H.eigenenergies()[idy]
#     print(str(round((idx + 1) / len(current) * 100, 2)) + "%")
#
# np.savetxt(path, energies)

#####################################################################################
energies = np.genfromtxt(path)
level_num = len(energies[0,:])
current = np.linspace(0.5,2,751)*1e-3
# for idx in range(3,level_num):
    # print(len(energies[:, idx]))
    # plt.plot(current*1e3, (energies[:,idx] - energies[:,2]), linestyle ='-', alpha = 0.5)

# for idx in range(2, level_num):
#     plt.plot(current * 1e3, energies[:, idx] - energies[:, 1], color='r', linestyle='--', alpha=1)
# for idx in range(3, level_num):
#     plt.plot(current * 1e3, energies[:, idx] - energies[:, 2], color='r', linestyle='--', alpha=1)
# for idx in range(4, level_num):
#     plt.plot(current * 1e3, energies[:, idx] - energies[:, 3], color='m', linestyle='-.', alpha=1)
# for idx in range(5, level_num):
#     plt.plot(current * 1e3, energies[:, idx] - energies[:, 4], color='m', linestyle='-.', alpha=1)

plt.plot(current * 1e3, energies[:, 1] - energies[:, 0], color='k', linestyle='-', alpha=1, label=r'$|00\rangle \rightarrow |10\rangle$')
plt.plot(current * 1e3, energies[:, 2] - energies[:, 0], color='k', linestyle='-', alpha=1, label=r'$|00\rangle \rightarrow |01\rangle$')
plt.plot(current * 1e3, energies[:, 3] - energies[:, 1], color='b', linestyle='--', alpha=1, label=r'$|10\rangle \rightarrow |11\rangle$')
plt.plot(current * 1e3, energies[:, 3] - energies[:, 2], color='b', linestyle='--', alpha=1, label=r'$|01\rangle \rightarrow |11\rangle$')
plt.xlabel('Current (mA)', size = 16.0)
plt.ylabel('Frequency (GHz)', size = 16.0)
plt.tick_params(labelsize = 16)
plt.ylim([0,12])
plt.show()