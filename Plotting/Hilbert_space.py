import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
from Fluxonium_hamiltonians.Single_small_junction import coupled_hamiltonian

# contrast_min = -1
# contrast_max = 1

#Qubit and computation parameters
Na = 25
E_l=0.5
E_c=1
E_j=5
g = 0.1

Nr = 2
wr = 7.5

phi_ext = np.linspace(0,1,201)
level_num = 15
energies = np.zeros((len(phi_ext),level_num))

# Compute eigensnergies
for idx, phi in enumerate(phi_ext):
    H = coupled_hamiltonian(Na, E_l, E_c, E_j, phi*2*np.pi, Nr, wr, g)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]
plt.title(Nr)
for idx in range(1,level_num):
    plt.plot(phi_ext, energies[:,idx]-energies[:,0], linewidth = '2',linestyle='--',color = 'k')

###########################################################################################
for idx, phi in enumerate(phi_ext):
    H = bare_hamiltonian(Na, E_l, E_c, E_j, phi*2*np.pi)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]
# plt.title('Na=', str(Na), 'Nr=', str(Nr))
for idx in range(1,level_num):
    plt.plot(phi_ext, energies[:,idx]-energies[:,0], linewidth = '2',linestyle='-', color = 'b')

plt.tick_params(labelsize = 16.0)
plt.ylim([0,20])
plt.xlim([0,1])
# directory = 'C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results'
# fname = "Coupled_fluxonium_spectrum_AugustusXVI_fit_20190906.txt"
# path = directory + '\\' + fname
# np.savetxt(path, energies)