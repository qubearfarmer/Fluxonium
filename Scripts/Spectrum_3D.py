import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
from Fluxonium_hamiltonians.Single_small_junction import charge_dispersive_shift as nChi

# contrast_min = -1
# contrast_max = 1

# Qubit and computation parameters
N = 50
E_l = 0.4
E_c = 0.84
E_j = 1.2
iState = 0
fState = 1
w = 7.56
g = 0.1

phi_ext = np.linspace(0.0, 0.55, 101)
level_num = 20
energies = np.zeros((len(phi_ext), level_num))
chi = np.zeros(len(phi_ext))
# Compute eigensnergies
for idx, phi in enumerate(phi_ext):
    H = bare_hamiltonian(N, E_l, E_c, E_j, phi * 2 * np.pi)
    for idy in range(level_num):
        energies[idx, idy] = H.eigenenergies()[idy]
        chi[idx] = nChi(N, level_num, E_l, E_c, E_j, phi * 2 * np.pi, iState, fState, w, g)
chi = abs(chi)
for idx, phi in enumerate(phi_ext):
    plt.plot(phi_ext[idx], energies[idx, 1] - energies[idx, 0], 'bo', alpha=chi[idx] / np.max(chi))
plt.show()
