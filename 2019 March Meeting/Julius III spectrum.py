import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

N = 30
#Julius III
# E_l=0.386
# E_c=1.18
# E_j=2.24

#Fluxonium in waveguide
# E_l=0.59
# E_c=1.17
# E_j=2.048

# #Qubit A
# E_l=1
# E_c=0.84
# E_j=3

#Qubit B
# E_l=1.14
# E_c=0.84
# E_j=4.86

# #Qubit C
# E_l=0.72
# E_c=0.84
# E_j=2.2
#
# #Qubit D
# E_l=0.52
# E_c=0.83
# E_j=2.2
#
# #Qubit E
# E_l=0.5
# E_c=0.86
# E_j=1.6
#
# #Qubit F
# E_l=0.41
# E_c=0.8
# E_j=3.4
#
# #Qubit G
# E_l=0.19
# E_c=1.14
# E_j=1.65
#
# #Qubit H
# E_l=0.79
# E_c=1
# E_j=4.43

# Augustus VI
# E_l=1.07
# E_c=1.03
# E_j=4.86
# E_l=1.51
# E_c=1.01
# E_j=5.24

# Augustus VII
# E_l=0.645
# E_c=1.03
# E_j=3.2
# E_l=0.875
# E_c=1.07
# E_j=2.93

# Augustus VIII
# E_l=0.6573854922452231
# E_c=1.051496337951542
# E_j=2.964238712618184
E_l=0.8878044224466595
E_c=0.9021660020559287
E_j=2.4888829123774214


phi_ext = np.linspace(0,1,101)
level_num = 20
energies = np.zeros((len(phi_ext),level_num))

# Compute eigensnergies
for idx, phi in enumerate(phi_ext):
    H = bare_hamiltonian(N, E_l, E_c, E_j, phi*2*np.pi)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

for idx in range(1,level_num):
    plt.plot(phi_ext, energies[:,idx]-energies[:,0], linewidth = 2.0, color = 'k')

for idx in range(2,level_num):
    plt.plot(phi_ext, energies[:,idx]-energies[:,1], linewidth = 2.0 ,linestyle = '--', color = 'b')

cavity_1 = np.ones(101)*7.5
cavity_2 = np.ones(101)*10
cavity_3 = np.ones(101)*12.8
# cavity_4 = np.ones(101)*12.48
# cavity_5 = np.ones(101)*13.11
# cavity_6 = np.ones(101)*16
plt.plot(phi_ext, cavity_1, linewidth = 3.0, color = 'r')
plt.plot(phi_ext, cavity_2, linewidth = 3.0, color = 'r')
plt.plot(phi_ext, cavity_3, linewidth = 3.0, color = 'r')
# plt.plot(phi_ext, cavity_4, linewidth = 3.0, color = 'r')
# plt.plot(phi_ext, cavity_5, linewidth = 3.0, color = 'r')
# plt.plot(phi_ext, cavity_6, linewidth = 3.0, color = 'r')
plt.ylim([0,20])
plt.tick_params(labelsize = 15.0)
plt.show()