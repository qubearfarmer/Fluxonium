from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import numpy as np
from matplotlib import pyplot as plt

contrast_min = -1
contrast_max = 1
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Two_tone'
measurement = '112117_Two_tone_spec_YOKO_0to1mA_Cav_7.3692GHz&-10dBm_QuBit2to10GHz&10dBm'
path = directory + '\\' + measurement
offset = 0.09
con_factor = 0.5/1.475

#Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
# current = current[1:-1]
# current = (current-offset)*con_factor
# freq = np.genfromtxt(path + '_FREQ.csv')
# freq = freq[1:]
# data = np.genfromtxt(path + '_PHASEMAG.csv')
# phase = data[1:,0] #phase is recorded in rad
# mag = data[1:,1]
# Z = np.zeros((len(current),len(freq)))
# for idx in range(len(current)):
#     temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
#     temp = temp*180/(np.pi)
#     # temp = mag[idx*len(freq):(idx+1)*len(freq)]
#     Z[idx,:] = temp - np.mean(temp)
# plt.figure(1)
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '112717_Two_tone_spec_YOKO_0to1mA_Cav_7.3692GHz&-10dBm_QuBit10to12GHz&10dBm'
# path = directory + '\\' + measurement
#
# #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
# current = current[1:-1]
# current = (current-offset)*con_factor
# freq = np.genfromtxt(path + '_FREQ.csv')
# freq = freq[1:]
# data = np.genfromtxt(path + '_PHASEMAG.csv')
# phase = data[1:,0] #phase is recorded in rad
# mag = data[1:,1]
# Z = np.zeros((len(current),len(freq)))
# for idx in range(len(current)):
#     temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
#     temp = temp*180/(np.pi)
#     # temp = mag[idx*len(freq):(idx+1)*len(freq)]
#     Z[idx,:] = temp - np.mean(temp)
# plt.figure(1)
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '112817_Two_tone_spec_YOKO_1to1.7mA_Cav_7.3692GHz&-10dBm_QuBit1to12GHz&10dBm'
# path = directory + '\\' + measurement
#
# #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
# current = current[1:-1]
# current = (current-offset)*con_factor
# freq = np.genfromtxt(path + '_FREQ.csv')
# freq = freq[1:]
# data = np.genfromtxt(path + '_PHASEMAG.csv')
# phase = data[1:,0] #phase is recorded in rad
# mag = data[1:,1]
# Z = np.zeros((len(current),len(freq)))
# for idx in range(len(current)):
#     temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
#     temp = temp*180/(np.pi)
#     # temp = mag[idx*len(freq):(idx+1)*len(freq)]
#     Z[idx,:] = temp - np.mean(temp)
# plt.figure(1)
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

#Qubit and computation parameters
N = 50
E_l = 0.5
E_c = 0.84
E_j = 3

phi_ext = np.linspace(-0.05,0.55,601)
level_num = 20
energies = np.zeros((len(phi_ext),level_num))

#Compute eigensnergies
for idx, phi in enumerate(phi_ext):
    H = bare_hamiltonian(N, E_l, E_c, E_j, phi*2*np.pi)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

#Plot eigensnergies
# fig1 = plt.figure(1)
# for idx in range(level_num):
#     plt.plot(phi_ext, energies[:,idx], linewidth = '2')
# plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
# plt.ylabel(r'Energy (GHz)')
# plt.ylim(top=30)
# plt.grid()

#Plot transition energies
# fig2 = plt.figure(2)

# for idx in range(1,level_num):
#     plt.plot(phi_ext, energies[:,idx]-energies[:,0], linewidth = '2', color = 'k')
plt.plot(phi_ext, energies[:,1]-energies[:,0], linewidth = 2 , linestyle = '-', color = 'k')
plt.plot(phi_ext, energies[:,2]-energies[:,0], linewidth = 2 , linestyle = '-', color = 'k')
plt.plot(phi_ext, energies[:,3]-energies[:,0], linewidth = 2 , linestyle = '-', color = 'k')
plt.plot(phi_ext, energies[:,4]-energies[:,0], linewidth = 2 , linestyle = '-', color = 'k')
plt.plot(phi_ext, energies[:,2]-energies[:,1], linewidth = 1.0 , linestyle = '--', color = 'b')
plt.plot(phi_ext, energies[:,3]-energies[:,1], linewidth = 1.0 , linestyle = '--', color = 'b')
plt.plot(phi_ext, energies[:,4]-energies[:,1], linewidth = 1.0 , linestyle = '--', color = 'b')
# plt.plot(phi_ext, energies[:,5]-energies[:,1], linewidth = 1.0 , linestyle = '--', color = 'r')
# plt.plot(phi_ext, energies[:,1]-energies[:,0]+7.369, linewidth = 1.5 , linestyle = '--', color = 'coral')
# plt.plot(phi_ext, energies[:,2]-energies[:,0]+7.369, linewidth = 1.5 , linestyle = '--', color = 'coral')
# plt.plot(phi_ext, energies[:,4]-energies[:,1]-3, linewidth = 1.0 , linestyle = '--', color = 'm')
plt.plot(phi_ext, (energies[:,1]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
plt.plot(phi_ext, (energies[:,2]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
plt.plot(phi_ext, (energies[:,3]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
plt.plot(phi_ext, (energies[:,4]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
plt.plot(phi_ext, (energies[:,5]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')

plt.plot(phi_ext, (energies[:,2]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
plt.plot(phi_ext, (energies[:,3]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
plt.plot(phi_ext, (energies[:,4]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
plt.plot(phi_ext, (energies[:,5]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
plt.plot(phi_ext, (energies[:,6]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,4]-energies[:,0])-7.369, linewidth = 1.5 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,5]-energies[:,0])-7.369, linewidth = 1.5 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,6]-energies[:,0])-7.369, linewidth = 1.5 , linestyle = '-.', color = 'm')
#
# plt.plot(phi_ext, (energies[:,5]-energies[:,2]), linewidth = 1.5 , linestyle = '-.', color = 'b')
# plt.plot(phi_ext, (energies[:,6]-energies[:,2]), linewidth = 1.5 , linestyle = '-.', color = 'b')
# plt.plot(phi_ext, (energies[:,4]-energies[:,2]), linewidth = 1.5 , linestyle = '-.', color = 'b')
# plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
# plt.ylabel(r'$\mathrm{E_i} - \mathrm{E_0}$')
plt.ylim([0,12])
plt.tick_params(labelsize = 18.0)
# plt.grid()

plt.show()
