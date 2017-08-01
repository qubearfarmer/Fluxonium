import numpy as np
from matplotlib import pyplot as plt
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import charge_dispersive_shift as nChi

#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Dispersive_shifts_wSquid"
path = directory + "\\" + simulation
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

current = np.linspace(0.035, 0.04, 501)
chi = np.zeros(len(current))
level_num = 30
energies = np.zeros((len(current),level_num))

iState = 0
fState = 1
B_coeff = 60
wr = 7.35
g = 0.092
kappa = 50
path = path+"_"+str(iState)+str(fState)+"_"+str(current[0]*1e3)+"to"+str(current[-1]*1e3)+"mA"
#######################################################################################################################
#Simulation part
'''
#Compute spectrum
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

#Dispersive shifts
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4

    chi[idx] = nChi(N, level_num, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState, wr, g)
np.savetxt(path+"_current.txt", current*1e3)
np.savetxt(path+"_energies.txt", energies)
np.savetxt(path+"_chi.txt", chi)
'''
#######################################################################################################################
#Plotting part
path = directory + "\\" + simulation
path = path+"_"+str(iState)+str(fState)+"_"+str(current[0]*1e3)+"to"+str(current[-1]*1e3)+"mA"
energies = np.genfromtxt(path+"_energies.txt")
chi = np.genfromtxt(path+"_chi.txt")
chi_angle = chi*1e3/(kappa/2) *180/np.pi

fig, ax1 = plt.subplots()
ax1.plot(current * 1e3, energies[:, fState] - energies[:, iState], color='k')
ax1.set_ylabel('Transition energy')
ax1.set_xlabel('Current (mA)')
for tl in ax1.get_yticklabels():
    tl.set_color('k')

ax2 = ax1.twinx()
ax2.plot(current * 1e3, chi_angle, 'g.')
ax2.set_ylabel('Dispersive shift (MHz)')
ax2.set_ylim([-5, 5])
# ax2.set_xlim([38.523, 38.75])
for t2 in ax2.get_yticklabels():
    t2.set_color('b')

ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
plt.grid()
plt.show()