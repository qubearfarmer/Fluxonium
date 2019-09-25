import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp_array as r_qp_array

# plt.rc('text', usetex=True)
# plt.rc('font', family='sans-serif')
#Define file directory

directory = "C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results"
fname = "Relaxation_blochnium_Zeus"
path = directory + "\\" + fname

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.626e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
#######################################################################################
N = 50
E_l = 0.203
E_c = 7.6
E_j = 7.25
level_num = 15
chain_num = 460
kB = 1.38e-23

iState = 0
fState = 1
phi_ext = np.linspace(0,0.5,101)
p_element = np.zeros(len(phi_ext))
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
gamma_cap = np.zeros(len(phi_ext))
gamma_cap_chain1 = np.zeros(len(phi_ext))
gamma_cap_chain2 = np.zeros(len(phi_ext))
gamma_flux = np.zeros(len(phi_ext))
gamma_qp = np.zeros(len(phi_ext))
gamma_qp_array = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))

#######################################################################################
# '''
# for idx, phi in enumerate(phi_ext):
#     p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
#     n_element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
#     qp_element[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2.0 * np.pi, iState, fState))
#     for idy in range(level_num):
#         energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2.0*np.pi).eigenenergies()[idy]
#
# np.savetxt(path + '_flux.txt', phi_ext)
# np.savetxt(path + '_energies.txt', energies)
# np.savetxt(path + '_chargeElement.txt', n_element)
# np.savetxt(path + '_fluxElement.txt', p_element)
# np.savetxt(path + '_qpElement.txt', qp_element)
# '''
#######################################################################################
phi_ext = np.genfromtxt(path+'_flux.txt')
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
w = energies[:,fState]-energies[:,iState]
T_diel = 1.0e-3
thermal_factor_diel = (1+np.exp(-h*w*1e9/(kB*T_diel)))
T_qp=1.0e-3
thermal_factor_qp = (1+np.exp(-h*w*1e9/(kB*T_qp)))
# C_chain = 36.0e-15
# Cg = 36.0e-18
#
for Q_cap in [0.2e6]:
    for idx in range(len(phi_ext)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
    label_text = 'Dielectric loss tangent = ' + str(Q_cap**-1)
    plt.semilogy(phi_ext, 1.0/gamma_cap *1e6, linewidth= 2.0, linestyle ='-',  label = label_text)
#
# for Q_cap in [1e3]:
#     for idx in range(len(phi_ext)):
#         gamma_cap_chain1[idx] = r_cap_chain1(C_chain, chain_num, Q_cap, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
#     plt.semilogy(phi_ext, 1.0/gamma_cap_chain1 *1e6, linewidth= 2.0, linestyle ='-')

# for Q_cap in [0.2e6]:
#     for idx in range(len(phi_ext)):
#         gamma_cap_chain2[idx] = r_cap_chain2(Cg, chain_num, Q_cap, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
#     plt.semilogy(phi_ext, 1.0/gamma_cap_chain2 *1e6, linewidth= 2.0, linestyle ='-')

for x_qp in [10e-7]:
    Q_qp = 1.0/x_qp
    for idx in range(len(phi_ext)):
        gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp, w[idx], qp_element[idx])
    # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
    label_text = 'Tunnel junction quasiparticle x_qp = ' + str(x_qp)
    plt.semilogy(phi_ext, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--', color='k', label = label_text)

# for T_qp in [0.25, 0.28]:
#     thermal_factor_qp = (1 + np.exp(-h * w * 1e9 / (kB * T_qp)))
#     for idx in range(len(phi_ext)):
#         gamma_qp[idx] = r_qp_finiteT(E_l, E_c, E_j, w[idx], qp_element[idx], T_qp)*thermal_factor_qp[idx]
#     # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='r')

for x_qp in [1e-6]:
    Q_qp = 1.0 / x_qp
    for idx in range(len(phi_ext)):
        gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx])*thermal_factor_qp[idx]
    label_text = 'Chain quasiparticle x_qp = ' + str(x_qp)
    plt.semilogy(phi_ext, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--', color = 'orange', label = label_text)
    # plt.semilogy(w, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')

plt.legend()
plt.xlabel(r'$\Phi/\Phi_o$')
plt.ylabel(r'$T_1 (\mu s)$')
plt.show()