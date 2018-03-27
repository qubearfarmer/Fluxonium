from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import numpy as np
from matplotlib import pyplot as plt

# contrast_min = -1
# contrast_max = 1

#Qubit and computation parameters
N = 50
E_l = 0.35
E_c = 0.85
E_j = 2.8

phi_ext = np.linspace(0.0,0.55,501)
level_num = 20
energies = np.zeros((len(phi_ext),level_num))

# Compute eigensnergies
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
fig2 = plt.figure(2)

# for idx in range(1,level_num):
#     plt.plot(phi_ext, energies[:,idx]-energies[:,0], linewidth = '2', color = 'k')
plt.plot(phi_ext, energies[:,1]-energies[:,0], linewidth = 2 , linestyle = '-', color = 'k')
plt.plot(phi_ext, energies[:,2]-energies[:,0], linewidth = 2 , linestyle = '-', color = 'k')
plt.plot(phi_ext, energies[:,3]-energies[:,0], linewidth = 2 , linestyle = '-', color = 'k')
# plt.plot(phi_ext, energies[:,4]-energies[:,0], linewidth = 2 , linestyle = '-', color = 'k')
plt.plot(phi_ext, energies[:,2]-energies[:,1], linewidth = 1.0 , linestyle = '--', color = 'b')
# plt.plot(phi_ext, energies[:,3]-energies[:,1], linewidth = 1.0 , linestyle = '--', color = 'b')
# plt.plot(phi_ext, energies[:,4]-energies[:,1], linewidth = 1.0 , linestyle = '--', color = 'b')
# plt.plot(phi_ext, energies[:,5]-energies[:,1], linewidth = 1.0 , linestyle = '--', color = 'r')
# plt.plot(phi_ext, energies[:,1]-energies[:,0]+7.369, linewidth = 1.5 , linestyle = '--', color = 'coral')
# plt.plot(phi_ext, energies[:,2]-energies[:,0]+7.369, linewidth = 1.5 , linestyle = '--', color = 'coral')
# plt.plot(phi_ext, energies[:,4]-energies[:,1]-3, linewidth = 1.0 , linestyle = '--', color = 'm')
# plt.plot(phi_ext, (energies[:,1]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
# plt.plot(phi_ext, (energies[:,2]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
# plt.plot(phi_ext, (energies[:,3]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
# plt.plot(phi_ext, (energies[:,4]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
# plt.plot(phi_ext, (energies[:,5]-energies[:,0])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'r')
#
# plt.plot(phi_ext, (energies[:,2]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,3]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,4]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,5]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,6]-energies[:,1])/2.0, linewidth = 1.0 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,4]-energies[:,0])-7.369, linewidth = 1.5 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,5]-energies[:,0])-7.369, linewidth = 1.5 , linestyle = '-.', color = 'm')
# plt.plot(phi_ext, (energies[:,6]-energies[:,0])-7.369, linewidth = 1.5 , linestyle = '-.', color = 'm')
#
# plt.plot(phi_ext, (energies[:,5]-energies[:,2]), linewidth = 1.5 , linestyle = '-.', color = 'b')
# plt.plot(phi_ext, (energies[:,6]-energies[:,2]), linewidth = 1.5 , linestyle = '-.', color = 'b')
# plt.plot(phi_ext, (energies[:,4]-energies[:,2]), linewidth = 1.5 , linestyle = '-.', color = 'b')
# plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
# plt.ylabel(r'$\mathrm{E_i} - \mathrm{E_0}$')
# plt.ylim([0,12])
# plt.tick_params(labelsize = 18.0)
# plt.grid()

# E_l = np.linspace(0.1,2,191)
# E_c = 1.7
# E_j = np.linspace(1,9,801)
# phi_ext = 0.5
# energies = np.zeros((len(E_l), len(E_j)))
# for idx_l in range(len(E_l)):
#     for idx_j in range(len(E_j)):
#         H = bare_hamiltonian(N, E_l[idx_l], E_c, E_j[idx_j], phi_ext*2*np.pi)
#         energies[idx_l,idx_j] = H.eigenenergies()[1]-H.eigenenergies()[0]
#         if energies[idx_l,idx_j] > 0.5 or energies[idx_l,idx_j] < 0.3:
#             energies[idx_l, idx_j] = 0
#
# directory = "C:\Data\Fluxonium #10 simulations"
# simulation = "Spectrum_hfq_Ec=1p7.txt"
# path = directory + '\\' +simulation
# np.savetxt(path, energies)
# X, Y = np.meshgrid(E_l,E_j)
# Z = np.genfromtxt(path).transpose()
# plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = 0.3, vmax = 0.5)

plt.show()
