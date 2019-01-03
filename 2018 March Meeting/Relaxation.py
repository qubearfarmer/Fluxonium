import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem

# plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')


#Define file directory
directory = "C:\\Users\\nguyen89\\Box\\Python Codes\\Fluxonium simulation results"
simulation = "Relaxation_22"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[7,5])
#######################################################################################
N = 50
E_l = 0.41
E_c = 0.8
E_j = 3.4
level_num = 15
T = 20e-3
iState = 0
fState = 1
phi_ext = np.linspace(0.0,0.5,501)
p_element = np.zeros(len(phi_ext))
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
gamma_cap = np.zeros(len(phi_ext))
gamma_ind = np.zeros(len(phi_ext))
gamma_qp = np.zeros(len(phi_ext))
gamma_qp_array = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))

#######################################################################################
for idx, phi in enumerate(phi_ext):
    p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
    n_element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
    qp_element[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2.0 * np.pi, iState, fState))
    for idy in range(level_num):
        energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2.0*np.pi).eigenenergies()[idy]

np.savetxt(path + '_energies.txt', energies)
np.savetxt(path + '_chargeElement.txt', n_element)
np.savetxt(path + '_fluxElement.txt', p_element)
np.savetxt(path + '_qpElement.txt', qp_element)

#######################################################################################
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
w = energies[:,fState]-energies[:,iState]
# fig, ax1 = plt.subplots(figsize=[10,7])
# ax2 = ax1.twinx()
# ax2.plot(phi_ext, w, color = 'black', linewidth = 2.0)
# plt.plot(phi_ext,w)

# for Q_cap in [0.5e6]:
#     for idx in range(len(phi_ext)):
#         gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap, w[idx], p_element[idx], T)
#     plt.semilogy(phi_ext, 1.0/gamma_cap *1e6, linewidth= 2.0)

# for x_qp in [1e-6]:l
#     Q_qp = 1.0/x_qp
#     for idx in range(len(phi_ext)):
#         gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp, w[idx], qp_element[idx], T)
#     plt.semilogy(phi_ext, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')

# for x_qp in [1e-8]:
#     Q_qp = 1.0 / x_qp
#     for idx in range(len(phi_ext)):
#         gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx], T)
#     plt.semilogy(phi_ext, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='-.')
#     plt.semilogy(phi_ext, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')

# ax2.tick_params(labelsize = 20.0)
# ax2.set_yticks([0,1,2,3,4])
# plt.tick_params(labelsize = 18.0)
# ax2.set_xticks([0,0.5])
# ax2.set_xlim([0,0.5])
# ax1.set_ylim([1e1,1e3])
# plt.ylabel("T1 (microseconds)", size = 18.0)
# plt.xlabel("$\Phi$", size = 18.0)
# plt.show()