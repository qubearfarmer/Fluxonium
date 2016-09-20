from Fluxonium_hamiltonians.Single_small_junction import coupled_hamiltonian
import numpy as np
from matplotlib import pyplot as plt

#Qubit and computation parameters
Na = 50
Nr = 10
E_l = 0.5
E_c = 2.5
E_j = 10
wr = 10
g = 0.1

phi_ext = np.linspace(-1,1,100)
level_num = 10
energies = np.zeros((len(phi_ext),level_num))

#Compute eigensnergies
for idx, phi in enumerate(phi_ext):
    H = coupled_hamiltonian(Na, E_l, E_c, E_j, phi*2*np.pi, Nr, wr, g)
    for idy in range(level_num):
        energies[idx,idy] = abs(H.eigenenergies()[idy])

#Plot eigensnergies
fig1 = plt.figure(1)
for idx in range(level_num):
    plt.plot(phi_ext, energies[:,idx], linewidth = '2')
plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
plt.ylabel(r'Energy (GHz)')
plt.ylim(top=30)

#Plot transition energies
fig2 = plt.figure(2)
for idx in range(1,level_num):
    plt.plot(phi_ext, energies[:,idx]-energies[:,0], linewidth = '2')
plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
plt.ylabel(r'$\mathrm{E_i} - \mathrm{E_0}$')
plt.ylim([0,30])

plt.show()