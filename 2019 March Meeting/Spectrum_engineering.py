import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

#Qubit and computation parameters
N = 30
E_c = 1.0
E_l_array = np.linspace(0.4,1.6,101)
E_j_array = np.linspace(3.0,5,101)

phi_ext = 0.5
level_num = 20
energies = np.zeros((len(E_l_array),len(E_j_array), level_num))

# Compute eigensnergies
for idx,E_l in enumerate(E_l_array):
    for idy, E_j in enumerate(E_j_array):
        H = bare_hamiltonian(N, E_l, E_c, E_j, phi_ext*2*np.pi)
        for idz in range(level_num):
            energies[idx,idy,idz] = H.eigenenergies()[idz]

directory = 'C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_SpectrumScan.txt"
path = directory + '\\' + fname
# np.savetxt(path,energies)

# spectrum = np.genfromtxt(path)
spectrum = energies
trans_energy_01 = spectrum[:,:,1] - spectrum[:,:,0]
trans_energy_21 = spectrum[:,:,2] - spectrum[:,:,1]
plt.figure(1, figsize =[6,6])
X,Y = np.meshgrid(E_l_array, E_j_array)
plt.pcolormesh(X,Y,(trans_energy_01 - 0.58), cmap= 'bwr', vmin = -0.5, vmax = 0.5)
# plt.xlabel('E_L')
# plt.ylabel('E_J')
plt.tick_params(labelsize=18)
plt.xlim([0.5,1.5])
plt.xticks([0.5,0.75,1,1.25,1.5])
plt.yticks([3,3.5,4, 4.5, 5])
plt.colorbar()
plt.figure(2, figsize =[6,6])
plt.pcolormesh(X,Y,(trans_energy_21-3.38), cmap= 'bwr', vmin = -0.5, vmax = 0.5)
plt.colorbar()
# plt.xlabel('E_L')
# plt.ylabel('E_J')
plt.tick_params(labelsize=18)
plt.xlim([0.5,1.5])
plt.xticks([0.5,0.75,1,1.25,1.5])
plt.yticks([3,3.5,4, 4.5, 5])
plt.show()
