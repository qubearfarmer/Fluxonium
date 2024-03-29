import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

N = 30
E_l=1
E_c=1
E_j=5
iState = 0
fState = 1
level_num = 10
phi_ext = np.linspace(0,0.5,101)
element = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))


# Compute eigensnergies
for idx, phi in enumerate(phi_ext):
    H = bare_hamiltonian(N, E_l, E_c, E_j, phi*2*np.pi)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]
freq = energies[:,1] - energies[:,0]
# for idx, phi in enumerate(phi_ext):
#     element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2*np.pi, iState, fState))
# fig1 = plt.figure(1)
# plt.plot(phi_ext, element)
for idx, phi in enumerate(phi_ext):
    element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2*np.pi, 0, 1))
plt.plot(phi_ext, element*freq)
# for idx, phi in enumerate(phi_ext):
#     element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2*np.pi, 0, 3))
# plt.plot(phi_ext, element)
# for idx, phi in enumerate(phi_ext):
#     element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2*np.pi, 1, 2))
# plt.plot(phi_ext, element, '--')
# for idx, phi in enumerate(phi_ext):
#     element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2*np.pi, 1, 3))
# plt.plot(phi_ext, element, '--')

# phi_ext = np.linspace(0,0.5,100)
# element = np.zeros(len(phi_ext))
# for idx, phi in enumerate(phi_ext):
#     element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2*np.pi, iState, fState))
# fig1 = plt.figure(1)
# # plt.semilogy(phi_ext, element**2, linewidth = '2')
# plt.plot(phi_ext, element**2, linewidth = '2')
plt.tick_params(labelsize = 18.0)
plt.show()
