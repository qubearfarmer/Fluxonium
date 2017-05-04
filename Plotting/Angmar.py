import numpy as np
from matplotlib import pyplot as plt
plt.rc('font', family='serif')

from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
N = 50
E_l = 0.5
E_c = 0.5
E_j = np.linspace(4, 12, 81)

phi_ext = 0.385
element01 = np.zeros(len(E_j))
element02 = np.zeros(len(E_j))
element13 = np.zeros(len(E_j))

for idx, Ej in enumerate(E_j):
    element01[idx]=abs(pem(N, E_l, E_c, Ej, phi_ext*2*np.pi, 0, 1))
    element02[idx] = abs(pem(N, E_l, E_c, Ej, phi_ext * 2 * np.pi, 0, 2))
    element13[idx] = abs(pem(N, E_l, E_c, Ej, phi_ext * 2 * np.pi, 1, 3))
plt.figure(figsize=[7,3.5])
plt.semilogy(E_j, element01, linewidth = 2.0, color = 'm')
plt.semilogy(E_j, element02, linewidth = 2.0, color = 'b')
plt.semilogy(E_j, element13, linewidth = 2.0, color = 'r')

plt.tick_params(labelsize = 24.0)
plt.xticks(np.linspace(4,12,3))
plt.yticks([1e-4, 1e-2, 1e0])
plt.show()
