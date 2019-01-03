import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import charge_matrix_element as nem
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem

#Define file directory
directory = "C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results"
fname = "MElements_wSquid"
path = directory + "\\" + fname

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

#Qubit and computation parameters
N = 50
E_l = 0.722729827116
E_c = 0.552669197076
E_j_sum = 17.61374383
A_j = 4.76321410213e-12
A_c = 1.50075181762e-10
d = 0.125005274368
beta_squid = 0.129912406349
beta_ext = 0.356925557542

B_coeff = 60
level_num = 5
current = np.linspace(0.045, 0.046, 101)
energies = np.zeros((len(current),level_num))
qp_element = np.zeros((len(current),2))
n_element = np.zeros(len(current))
p_element = np.zeros(len(current))

iState = 0
fState = 1

path = path+'_'+str(iState)+'to'+str(fState)+'_from_' + str(current[0]*1e3) +'to'+ str(current[-1]*1e3) +'mA'
########################################################################################################################
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx, idy] = H.eigenenergies()[idy]
    n_element [idx] = nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    p_element[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element[idx,:] = qpem(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)

np.savetxt(path+'_energies.txt', energies)
np.savetxt(path+'_chargeElement.txt', n_element)
np.savetxt(path+'_phaseElement.txt', p_element)
np.savetxt(path+'_qpElement.txt', qp_element)
########################################################################################################################
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_phaseElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
# trans_energy = energies[:, 1] - energies[:, 0]
# fig, ax1 = plt.subplots()
# ax1.plot(current * 1e3, trans_energy, color='b', linewidth='2')
# trans_energy = energies[:, 2] - energies[:, 0]
# ax1.plot(current * 1e3, trans_energy, color='b', linewidth='2')
# trans_energy = energies[:, 3] - energies[:, 0]
# ax1.plot(current * 1e3, trans_energy, color='b', linewidth='2')
# trans_energy = energies[:, 4] - energies[:, 0]
# ax1.plot(current * 1e3, trans_energy, color='b', linewidth='2')
# trans_energy = energies[:, 2] - energies[:, 1]
# ax1.plot(current * 1e3, trans_energy, color='r', linewidth='2')
# trans_energy = energies[:, 3] - energies[:, 1]
# ax1.plot(current * 1e3, trans_energy, color='r', linewidth='2')
# trans_energy = energies[:, 4] - energies[:, 1]
# ax1.plot(current * 1e3, trans_energy, color='r', linewidth='2')
# ax1.set_ylabel('Transition energy')
# ax1.set_xlabel('Current (mA)')
# for tl in ax1.get_yticklabels():
#     tl.set_color('k')

# ax2 = ax1.twinx()
# ax2.plot(current * 1e3, n_element, 'm--')
# # ax2.plot(current*1e3, qp_element[:,0]**2 + qp_element[:,1]**2, 'g--')
# ax2.set_ylabel('Matrix element')
# ax2.set_ylim([0.0, 1])
# # ax1.set_ylim([2.5,3.5])
# for t2 in ax2.get_yticklabels():
#     t2.set_color('b')
# ax1.tick_params(labelsize=18)
# ax2.tick_params(labelsize=18)


plt.plot(current, p_element)
#plt.plot(current, qp_element[:,1])
plt.grid()
plt.show()
