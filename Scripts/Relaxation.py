from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp_array as r_qp_array
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap_Z as r_cap_Z
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_ind as r_ind

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.figure(figsize=[10,7])
#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Relaxation"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[20,10])
#######################################################################################
N = 50
E_l = 1.0168
E_c = 0.835
E_j = 2.996
level_num = 15

iState = 0
fState = 1
phi_ext = np.linspace(-0.05,0.55,601)
p_element = np.zeros(len(phi_ext))
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
gamma_cap = np.zeros(len(phi_ext))
gamma_ind = np.zeros(len(phi_ext))
gamma_qp = np.zeros(len(phi_ext))
gamma_qp_array = np.zeros(len(phi_ext))
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
'''
#######################################################################################
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
w = energies[:,fState]-energies[:,iState]

for Q_cap in [0.5e6]:
    for idx in range(len(phi_ext)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap, w[idx], p_element[idx])
    plt.semilogy(phi_ext*1.53/0.5, 1.0/gamma_cap *1e6, linewidth= 2.0, linestyle ='-')

# for x_qp in [4e-8]:
#     Q_qp = 1.0 / x_qp
#     for idx in range(len(phi_ext)):
#         gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx])
#     plt.semilogy(phi_ext*1.53/0.5, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--')

for x_qp in [2e-6, 3e-6, 5e-6]:
    Q_qp = 1.0/x_qp
    for idx in range(len(phi_ext)):
        gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp, w[idx], qp_element[idx])
    # plt.semilogy(phi_ext * 1.53 / 0.5, 1.0 / (gamma_qp) * 1e6, linewidth=2.5, linestyle='--')
    plt.semilogy(phi_ext*1.53/0.5, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.5, linestyle='-.')


# plt.ylim([1e1,1e4])
# plt.yticks([5e2,1e3,2e3,5e3])
# plt.xlabel(r'$\Phi_e/\Phi_o$', fontsize = 18)
# plt.xticks([0,0.5])
# plt.ylabel(r'$T_1/Q$', fontsize = 18)

############################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Summary'
fname = '20170825_T1.txt'
path = directory + '\\'+ fname
data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
current = data[:,0]
freq = data[:,1]
T1 = data[:,2]
plt.errorbar(current, T1, fmt='s', mfc='none', mew=2.0, mec='blue')
############################################################################

plt.tick_params(labelsize=18)
plt.show()
