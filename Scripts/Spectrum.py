from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import numpy as np
from matplotlib import pyplot as plt

#Qubit and computation parameters
N = 50
E_l = 0.5
E_c = 0.86
E_j = 1.574

phi_ext = np.linspace(-0.05,0.55,101)
level_num = 10
energies = np.zeros((len(phi_ext),level_num))

#Compute eigensnergies
for idx, phi in enumerate(phi_ext):
    H = bare_hamiltonian(N, E_l, E_c, E_j, phi*2*np.pi)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

#Plot eigensnergies
fig1 = plt.figure(1)
for idx in range(level_num):
    plt.plot(phi_ext, energies[:,idx], linewidth = '2')
plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
plt.ylabel(r'Energy (GHz)')
plt.ylim(top=30)
plt.grid()

#Plot transition energies
fig2 = plt.figure(2)
for idx in range(1,level_num):
    plt.plot(phi_ext, energies[:,idx]-energies[:,0], linewidth = '2')
plt.xlabel(r'$\varphi_\mathrm{ext}/2\pi$')
plt.ylabel(r'$\mathrm{E_i} - \mathrm{E_0}$')
plt.ylim([0,30])
plt.grid()

plt.show()
