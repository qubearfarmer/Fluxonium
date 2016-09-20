from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian_alt
import numpy as np
from matplotlib import pyplot as plt
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

#Qubit and computation parameters
N = 50
E_l = 0.5
E_c = 2.5
E_j_sum = 20

N = 50
E_l = 0.746959655208
E_c = 0.547943694372
E_j_sum = 21.9627179709
level_num = 10
B_coeff = 60
A_j = 3.80888914574e-12
A_c = 1.49982268962e-10
beta_squid = 0.00378012644185
beta_ext = 0.341308382441
d=0.0996032153487
current = np.linspace(0,0.05,1000)
energies = np.zeros((len(current),level_num))

#Compute eigenenergies
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

#Plot transition energies
for idx in range(1, level_num):
    plt.plot(current, energies[:,idx]-energies[:,0])

#Alternate Hamiltonian
# for idx, curr in enumerate(current):
#     flux_squid = curr*B_coeff*A_j*1e-4
#     flux_ext = curr*B_coeff*A_c*1e-4
#     H = bare_hamiltonian_alt(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
#                          2 * np.pi * (flux_ext / phi_o - beta_ext))
#     for idy in range(level_num):
#         energies[idx,idy] = H.eigenenergies()[idy]
# for idx in range(1, level_num):
#     plt.plot(current, energies[:,idx]-energies[:,0])

plt.show()
