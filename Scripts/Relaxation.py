from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_ind as r_ind

import numpy as np
from matplotlib import pyplot as plt

#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Relaxation"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
plt.figure(figsize=[20,10])
#######################################################################################
N = 50
E_l = 0.525
E_c = 2.5
E_j = 8.9
level_num = 10

iState = 0
fState = 2
phi_ext = np.linspace(-0.03,0.03,61)
p_element = np.zeros(len(phi_ext))
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
gamma_cap = np.zeros(len(phi_ext))
gamma_ind = np.zeros(len(phi_ext))
gamma_qp = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))
'''
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
######################################################################################
'''
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
w = energies[:,fState]-energies[:,iState]

######################################################################################
######################################################################################
iState = 1
fState = 2
p_element12 = np.zeros(len(phi_ext))
n_element12 = np.zeros(len(phi_ext))
qp_element12 = np.zeros(len(phi_ext))
gamma_cap12 = np.zeros(len(phi_ext))
gamma_ind12 = np.zeros(len(phi_ext))
gamma_qp12 = np.zeros(len(phi_ext))
'''
#######################################################################################
for idx, phi in enumerate(phi_ext):
    p_element12[idx]=abs(pem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
    n_element12[idx]=abs(nem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
    qp_element12[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2.0 * np.pi, iState, fState))

np.savetxt(path + '_chargeElement12.txt', n_element12)
np.savetxt(path + '_fluxElement12.txt', p_element12)
np.savetxt(path + '_qpElement12.txt', qp_element12)
######################################################################################
'''
n_element12 = np.genfromtxt(path+'_chargeElement12.txt')
p_element12 = np.genfromtxt(path+'_fluxElement12.txt')
qp_element12 = np.genfromtxt(path+'_qpElement12.txt')
w12 = energies[:,fState]-energies[:,iState]

for Q_cap in [9e6]:
    for idx in range(len(phi_ext)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, 6e6, w[idx], p_element[idx])
        gamma_cap12[idx] = r_cap(E_l, E_c, E_j, Q_cap, w12[idx], p_element12[idx])
    plt.semilogy(phi_ext, 1.0/(gamma_cap+gamma_cap12)*1e6, linewidth ='2')
    plt.semilogy(phi_ext, 1.0 / (gamma_cap) * 1e6, linewidth='2', linestyle ='--')
    plt.semilogy(phi_ext, 1.0 / (gamma_cap12) * 1e6, linewidth='2', linestyle='--')

# for Q_capx in [3e6]:
#     Q_cap = np.zeros(len(phi_ext))
#     Q_cap12 = np.zeros(len(phi_ext))
#     for idx in range(len(phi_ext)):
#         Q_cap[idx] = Q_capx*(5/w[idx])**(0.7)
#         Q_cap12[idx] = Q_capx * (5 / w12[idx]) ** (0.7)
#         gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap[idx], w[idx], p_element[idx])
#         gamma_cap12[idx] = r_cap(E_l, E_c, E_j, Q_cap12[idx], w12[idx], p_element12[idx])
#     plt.semilogy(phi_ext, 1.0/(gamma_cap+gamma_cap12)*1e6, linewidth ='2')

# for Q_ind in [1e5, 1e6, 1e7, 1e8, 5e8]:
#     for idx in range(len(phi_ext)):
#         gamma_ind[idx] = r_ind(E_l, E_c, E_j, Q_ind, w[idx], p_element[idx])
#     plt.semilogy(phi_ext, 1.0/gamma_ind*1e6, linewidth ='2')

for Q_qp in [10e6]:
    for idx in range(len(phi_ext)):
        gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp, w[idx], qp_element[idx])
        gamma_qp12[idx] = r_qp(E_l, E_c, E_j, 1200e6, w12[idx], qp_element12[idx])
    plt.semilogy(phi_ext, 1.0/(gamma_qp)*1e6, linewidth ='2',linestyle='-.')
    plt.semilogy(phi_ext, 1.0 / (gamma_qp12) * 1e6, linewidth='2',linestyle='-.')


plt.ylim([4e2,6e3])
plt.yticks([5e2,1e3,2e3,5e3])
plt.tick_params(labelsize=18)
plt.show()
