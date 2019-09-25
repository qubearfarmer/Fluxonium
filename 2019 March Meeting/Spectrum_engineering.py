import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

#Qubit and computation parameters
N = 30
E_c = 1.0
E_la = 1.0
E_lb = 1.3
E_ja_array = np.linspace(2.5,6.5,81)
E_jb_array = np.linspace(2.5,6.5,101)

phi_ext = 0.5
level_num = 20
delta_energies = np.zeros((len(E_ja_array),len(E_jb_array), 2))

# Compute eigensnergies
for idx,E_ja in enumerate(E_ja_array):
    H = bare_hamiltonian(N, E_la, E_c, E_ja, phi_ext * 2 * np.pi)
    eigenenergies_a = H.eigenenergies()
    w01a = eigenenergies_a[1] - eigenenergies_a[0]
    w21a = eigenenergies_a[2] - eigenenergies_a[1]
    for idy, E_jb in enumerate(E_jb_array):
        H = bare_hamiltonian(N, E_lb, E_c, E_jb, phi_ext*2*np.pi)
        eigenenergies_b = H.eigenenergies()
        w01b = eigenenergies_b[1] - eigenenergies_b[0]
        w21b = eigenenergies_b[2] - eigenenergies_b[1]
        delta_energies[idx, idy, 0] = w01b - w01a
        delta_energies[idx, idy, 1] = w21b - w21a

directory = 'C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_SpectrumScan.txt"
path = directory + '\\' + fname
# np.savetxt(path,energies)

# spectrum = np.genfromtxt(path)
plt.figure(1, figsize =[6,6])
X,Y = np.meshgrid(E_ja_array, E_jb_array)
plt.pcolormesh(X,Y,delta_energies[:, :, 0].transpose(), cmap= 'bwr', vmin = -0.5, vmax = 0.5)
plt.xlabel('E_Ja',size=18)
plt.ylabel('E_Jb',size=18)
plt.tick_params(labelsize=16)
plt.title(str(E_lb))
# plt.xlim([0.5,1.5])
# plt.xticks([0.5,0.75,1,1.25,1.5])
# plt.yticks([2.5,3,3.5,4, 4.5, 5])
# plt.colorbar()
plt.figure(2, figsize =[6,6])
plt.pcolormesh(X,Y,delta_energies[:, :, 1].transpose(), cmap= 'bwr', vmin = -0.5, vmax = 0.5)
# plt.colorbar()
plt.xlabel('E_Ja',size=18)
plt.ylabel('E_Jb',size=18)
plt.title(str(E_lb))
plt.tick_params(labelsize=16)
# plt.xlim([0.5,1.5])
# plt.xticks([0.5,0.75,1,1.25,1.5])
# plt.yticks([2.5,3,3.5,4, 4.5, 5])
plt.show()