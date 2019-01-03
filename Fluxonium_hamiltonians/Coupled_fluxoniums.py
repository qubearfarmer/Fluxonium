import numpy as np
from matplotlib import pyplot as plt
from qutip import *

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.626e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

directory = "C:\\Users\\nguyen89\\Box\\Python Codes\\Fluxonium simulation results"
fname = "Coupled_fluxonium_spectrum.txt"
path = directory + '\\' + fname

N1 = 20
E_l1 = 0.7
E_c1 = 1.0
E_j1 = 4.5

N2 = 20
E_l2 = 0.7
E_c2 = 1.0
E_j2 = 4

J_c = 0.3
level_num = 20
phi_ext = np.linspace(0, 0.5, 101)
spectrum = np.zeros((len(phi_ext), level_num))
nem = np.zeros((len(phi_ext), level_num * 2))

for idx, phi in enumerate(phi_ext):

    a1 = tensor(destroy(N1), qeye(N2))
    phi1 = (a1 + a1.dag()) * (8.0 * E_c1 / E_l1) ** (0.25) / np.sqrt(2.0)
    na1 = 1.0j * (a1.dag() - a1) * (E_l1 / (8 * E_c1)) ** (0.25) / np.sqrt(2.0)
    ope1 = 1.0j * (phi1 - 2 * np.pi * phi)
    H1 = 4.0 * E_c1 * na1 ** 2.0 + 0.5 * E_l1 * phi1 ** 2.0 - 0.5 * E_j1 * (ope1.expm() + (-ope1).expm())

    a2 = tensor(qeye(N1), destroy(N2))
    phi2 = (a2 + a2.dag()) * (8.0 * E_c2 / E_l2) ** (0.25) / np.sqrt(2.0)
    na2 = 1.0j * (a2.dag() - a2) * (E_l2 / (8 * E_c2)) ** (0.25) / np.sqrt(2.0)
    ope2 = 1.0j * (phi2 - 2 * np.pi * phi)
    H2 = 4.0 * E_c2 * na2 ** 2.0 + 0.5 * E_l2 * phi2 ** 2.0 - 0.5 * E_j2 * (ope2.expm() + (-ope2).expm())

    Hc = J_c * na1 * na2
    H = H1 + H2 + Hc

    eigenenergies, eigenstates = H.eigenstates()
    for idy in range(level_num):
        spectrum[idx, idy] = eigenenergies[idy]
    nem[idx, 0] = abs(na1.matrix_element(eigenstates[0], eigenstates[1]))
    nem[idx, 1] = abs(na2.matrix_element(eigenstates[0], eigenstates[1]))
    nem[idx, 2] = abs(na1.matrix_element(eigenstates[0], eigenstates[2]))
    nem[idx, 3] = abs(na2.matrix_element(eigenstates[0], eigenstates[2]))
    nem[idx, 4] = abs(na1.matrix_element(eigenstates[0], eigenstates[3]))
    nem[idx, 5] = abs(na2.matrix_element(eigenstates[0], eigenstates[3]))
    nem[idx, 6] = abs(na1.matrix_element(eigenstates[0], eigenstates[4]))
    nem[idx, 7] = abs(na2.matrix_element(eigenstates[0], eigenstates[4]))
    print(str(round((idx + 1) / len(phi_ext) * 100, 2)) + "%")

directory = "C:\\Users\\nguyen89\\Box\\Python Codes\\Fluxonium simulation results"
fname = "Coupled_fluxonium_spectrum.txt"
path = directory + '\\' + fname
# np.savetxt(path, spectrum)
energy = np.genfromtxt(path)

fname = "Coupled_fluxonium_nem.txt"
path = directory + '\\' + fname
# np.savetxt(path, nem)
n_me = np.genfromtxt(path)
#############################################################

plt.figure(1)
plt.plot(phi_ext, energy[:, 1] - energy[:, 0])
plt.plot(phi_ext, energy[:, 2] - energy[:, 0])
plt.plot(phi_ext, energy[:, 3] - energy[:, 0])
plt.plot(phi_ext, energy[:, 4] - energy[:, 0])
plt.plot(phi_ext, energy[:, 5] - energy[:, 0])
plt.plot(phi_ext, energy[:, 6] - energy[:, 0])
plt.plot(phi_ext, energy[:, 7] - energy[:, 0])
plt.plot(phi_ext, energy[:, 8] - energy[:, 0])
plt.plot(phi_ext, energy[:, 9] - energy[:, 0])
# plt.plot(phi_ext, energy[:,2]-energy[:,1], '--')
# plt.plot(phi_ext, energy[:,3]-energy[:,1], '--')
# plt.plot(phi_ext, energy[:,4]-energy[:,1], '--')
# plt.plot(phi_ext, energy[:,5]-energy[:,1], '--')
# plt.plot(phi_ext, energy[:,6]-energy[:,1], '--')
plt.figure(2)
plt.plot(phi_ext, n_me[:, 0])
# plt.plot(phi_ext, n_me[:,1])
# plt.plot(phi_ext, n_me[:,2])
plt.plot(phi_ext, n_me[:, 3])
# plt.plot(phi_ext, nem[:,4])
# plt.plot(phi_ext, nem[:,5])
# plt.plot(phi_ext, nem[:,6])
# plt.plot(phi_ext, nem[:,7])
plt.show()
