import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
plt.rc('font', family='serif')
rc('text', usetex=False)
from Fluxonium_hamiltonians.Squid_small_junctions import coupled_hamiltonian
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Dispersive_shifts_wSquid"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[20,5])
#Qubit and computation parameters
N = 40
Nr = 10
E_l=0.735773762652
E_c=0.537375025825
E_j_sum=27
A_j=3.83424869313e-12
A_c=1.46689233147e-10
d=0.185865262485
beta_squid=-2.58488114861e-05
beta_ext=-0.0251115059548
B_coeff = 60
g=0.09
wr = 10.304
level_num = 5

current = np.linspace(0, 0.05, 501)
energies = np.zeros((len(current),level_num))
sim_dat = np.zeros((len(current),3))
#Compute eigenenergies
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = coupled_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), Nr, wr, g)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

np.savetxt(path + "_energies.txt", energies)
energies = np.genfromtxt(path + "_energies.txt")

#Plot transition energies
for idx in range(1, level_num):
    plt.plot(current*1e3, energies[:,idx]-energies[:,0], 'k-')

plt.show()