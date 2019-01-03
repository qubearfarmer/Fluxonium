import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import coupled_hamiltonian

directory = "C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results"
simulation = "Spectrum_coupled_#28"
path = directory + "\\" + simulation

#Qubit and computation parameters
Na = 30
Nr = 5
E_l = 1.125
E_c = 0.847
E_j = 4.79
wr = 7.3375
g = 0.1

phi_ext = np.linspace(0.58,0.584,11)
level_num = 7
energies = np.zeros((len(phi_ext),level_num))

########################################################################################
for idx, phi in enumerate(phi_ext):
    H = coupled_hamiltonian(Na, E_l, E_c, E_j, phi*2*np.pi, Nr, wr, g)
    for idy in range(level_num):
        energies[idx,idy] = abs(H.eigenenergies()[idy])
np.savetxt(path + '_energies.txt', energies)
########################################################################################
energies = np.genfromtxt(path+'_energies.txt')

#Plot eigensnergies
# fig1 = plt.figure(1)
# for idx in range(level_num):
#     plt.plot(phi_ext, energies[:,idx], linewidth = '2')
# plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
# plt.ylabel(r'Energy (GHz)')
# plt.ylim(top=30)

#Plot transition energies
# fig2 = plt.figure(2)
for idx in range(3,5):
    plt.plot(phi_ext, energies[:,idx]-energies[:,0], linewidth = '2', color = 'b')
# plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
# plt.ylabel(r'$\mathrm{E_i} - \mathrm{E_0}$')
# plt.ylim([0,30])

plt.show()