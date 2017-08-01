import numpy as np
from matplotlib import pyplot as plt
from Fluxonium_hamiltonians.Squid_small_junctions import charge_matrix_element as nem
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian as H
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_ind as r_ind

#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Relaxation_wSquid"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34     #Placnk's constant
phi_o = h/(2*e)  #Flux quantum

#######################################################################################
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
current = np.linspace(0.045, 0.047, 201)
level_num = 5
B_coeff = 60

n_element = np.zeros(len(current))
p_element = np.zeros(len(current))
qp_element = np.zeros((len(current),2))
energies = np.zeros((len(current),level_num))

iState = 0
fState = 1
path = path+'_'+str(iState)+'to'+str(fState)+'_from_' + str(current[0]*1e3) +'to'+ str(current[-1]*1e3) +'mA'
'''
#######################################################################################
for idx, curr in enumerate(current):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    n_element[idx]=abs(nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState))
    p_element[idx] = abs(pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState))
    qp_element[idx,:] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    for idy in range(level_num):
        energies[idx,idy] = H(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),2 * np.pi * (flux_ext / phi_o - beta_ext)).eigenenergies()[idy]

np.savetxt(path+'_energies.txt', energies)
np.savetxt(path+'_chargeElement.txt', n_element)
np.savetxt(path+'_fluxElement.txt', p_element)
np.savetxt(path+'_qpElement.txt', qp_element)
#######################################################################################
'''
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
w = energies[:,fState] - energies[:,iState]

gamma_cap = np.zeros(len(current))
gamma_qp = np.zeros((len(current),2))
for Q_cap in [1e5, 5e5, 1e6, 2e6]:
    for idx in range(len(current)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j_sum, d, Q_cap, w[idx], p_element[idx])
    plt.semilogy(current, 1.0 / (gamma_cap) * 1e6, linewidth='2', linestyle ='-')

#for Q_qp in [5e6]:
#    for idx in range(len(current)):
#        gamma_qp[idx,:] = r_qp(E_l, E_c, E_j_sum, d, Q_qp, w[idx], qp_element[idx,:])
#    plt.semilogy(current, 1.0 / (gamma_qp[:,0] + gamma_qp[:,1]) * 1e6, linewidth='2', linestyle ='-')

plt.show()
