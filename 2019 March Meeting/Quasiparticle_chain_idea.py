import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp_array as r_qp_array
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H

###########################################
N = 40
E_j = 3.55
E_c = 1.04
E_l_array = np.linspace(0.2, 1.2, 101)
phi = 0.5

level_num = 15
kB = 1.38e-23

iState = 0
fState = 1

p_element = np.zeros(len(E_l_array))
n_element = np.zeros(len(E_l_array))
qp_element = np.zeros(len(E_l_array))
gamma_cap = np.zeros(len(E_l_array))
gamma_cap_chain1 = np.zeros(len(E_l_array))
gamma_cap_chain2 = np.zeros(len(E_l_array))
gamma_flux = np.zeros(len(E_l_array))
gamma_qp = np.zeros(len(E_l_array))
gamma_qp_array = np.zeros(len(E_l_array))
energies = np.zeros((len(E_l_array),level_num))



for idx, E_l in enumerate(E_l_array):
    p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
    for idy in range(level_num):
        energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2.0*np.pi).eigenenergies()[idy]
w = energies[:,fState]-energies[:,iState]

for x_qp in [1e-8, 5e-8, 10e-8]:
    Q_qp = 1.0 / x_qp
    for idx in range(len(E_l_array)):
        gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx])
    plt.plot(E_l_array, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--')#, color = 'orange')
    # plt.semilogy(w, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')

plt.yscale('log')
plt.xlabel ('E_L (GHz)')
plt.ylabel('T1 (us)')
#######################################################################

# N = 40
# E_l = 1
# E_c = 1.0
# E_j_array = np.linspace(2.0, 5.0, 201)
# phi = 0.5
#
# level_num = 15
# kB = 1.38e-23
#
# iState = 0
# fState = 1
#
# p_element = np.zeros(len(E_j_array))
# n_element = np.zeros(len(E_j_array))
# qp_element = np.zeros(len(E_j_array))
# gamma_cap = np.zeros(len(E_j_array))
# gamma_cap_chain1 = np.zeros(len(E_j_array))
# gamma_cap_chain2 = np.zeros(len(E_j_array))
# gamma_flux = np.zeros(len(E_j_array))
# gamma_qp = np.zeros(len(E_j_array))
# gamma_qp_array = np.zeros(len(E_j_array))
# energies = np.zeros((len(E_j_array),level_num))
#
#
#
# for idx, E_j in enumerate(E_j_array):
#     p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
#     for idy in range(level_num):
#         energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2.0*np.pi).eigenenergies()[idy]
# w = energies[:,fState]-energies[:,iState]
#
# for x_qp in [1e-8, 5e-8, 10e-8]:
#     Q_qp = 1.0 / x_qp
#     for idx in range(len(E_j_array)):
#         gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx])
#     plt.plot(E_j_array, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--')#, color = 'orange')
#     # plt.semilogy(w, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')

# plt.yscale('log')
# plt.xlabel ('E_J (GHz)')
# plt.ylabel('T1 (us)')

plt.show()