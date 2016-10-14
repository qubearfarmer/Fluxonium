from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
import numpy as np
from matplotlib import pyplot as plt

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

#######################################################################################
N = 50
E_l = 0.5
E_c = 2.5
E_j = 10
level_num = 10
iState = 0
fState = 1

phi_ext = np.linspace(0,0.5,100)
p_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))

#######################################################################################
for idx, phi in enumerate(phi_ext):
    p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2*np.pi, iState, fState))
    qp_element[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2 * np.pi, iState, fState))
    for idy in range(level_num):
        energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[idy]

trans_energy = energies[:,fState]-energies[:,iState]
#######################################################################################
Q_cap = 3e6
Q_ind = 500e6
Q_qp = 0.3e6
cap = e**2/(2*E_c)
cap = e**2/(2*E_c)
Y_cap = trans_energy*2*np.pi*cap

fig, ax1 = plt.subplots()
ax1.plot(phi_ext, trans_energy, color = 'k', linewidth = '2')
ax1.set_ylabel('Transition energy')
ax1.set_xlabel('Ext flux')
for tl in ax1.get_yticklabels():
    tl.set_color('k')

ax2 = ax1.twinx()
ax2.plot(phi_ext, qp_element, 'b--')
ax2.set_ylabel('Charge matrix element')
# ax2.set_ylim([-0.5,0.5])
for t2 in ax2.get_yticklabels():
    t2.set_color('b')
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)

plt.show()
