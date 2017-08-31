import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
plt.rc('font', family='serif')
rc('text', usetex=False)
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[20,5])
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
B_coeff = 95.75
level_num = 5
current = np.linspace(0.0277, 0.029, 501)
energies = np.zeros((len(current),level_num))
sim_dat = np.zeros((len(current),3))
iState = 0
fState = 1
#Compute eigenenergies
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

plt.plot(current * 1e3, energies[:, 1] - energies[:, 0], 'k-')
plt.plot(current*1e3, energies[:,2]-energies[:,0], 'k-')
plt.plot(current*1e3, energies[:,3]-energies[:,0], 'k-')
plt.plot(current*1e3, energies[:,2]-energies[:,1], 'b--')

#Plot transition energies
# for idx in range(1, level_num):
#     plt.plot(current*1e3, energies[:,idx]-energies[:,0], 'k-')
# for idx in range(2, level_num):
#     plt.plot(current*1e3, energies[:,idx]-energies[:,1], 'r-')

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

##############################################################################################
#Measured spectrum
#01 transition
# directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Corrected flux"
#
# simulation = "T1avg(0to1)vs flux 41p52 to 42p0mA.csv"
# path = directory + "\\" + simulation
# data = np.genfromtxt(path, delimiter =',',dtype=float)
# plt.plot(data[1::,0], data[1::,1], 'ro')
#
# simulation = "T1_T2_qubit f(0to1)vs flux 43p65_45p4mA.csv"
# path = directory + "\\" + simulation
# data = np.genfromtxt(path, delimiter =',',dtype=float)
# plt.plot(data[1::,0], data[1::,1], 'ro')
#
# simulation = "T1 avg_T2_qubit f(0to1) vs flux_39p46 to 39p39mA.csv"
# path = directory + "\\" + simulation
# data = np.genfromtxt(path, delimiter =',',dtype=float)
# plt.plot(data[1::,0], data[1::,1], 'ro')
#
# simulation = "T1 avg_T2_qubit f(0to1) vs flux_38p5 to 38p76mA.csv"
# path = directory + "\\" + simulation
# data = np.genfromtxt(path, delimiter =',',dtype=float)
# plt.plot(data[1::,0], data[1::,1], 'ro')
#
# #02 transition
# simulation = "T1 avg_T2_qubit f(0to2) vs flux_38p76 to 38p26mA.csv"
# path = directory + "\\" + simulation
# data = np.genfromtxt(path, delimiter =',',dtype=float)
# plt.plot(data[1::,0], data[1::,1], 'ro')
#
# simulation = "T1 avg_T2_qubit f(0to2) vs flux_39p37 to 39p78mA.csv"
# path = directory + "\\" + simulation
# data = np.genfromtxt(path, delimiter =',',dtype=float)
# plt.plot(data[1::,0], data[1::,1], 'ro')
#
# simulation = "T1 avg_T2_qubit f(0to2) vs flux_41p5 to 42mA.csv"
# path = directory + "\\" + simulation
# data = np.genfromtxt(path, delimiter =',',dtype=float)
# plt.plot(data[1::,0], data[1::,1], 'ro')
##############################################################################################
#Saving data
# directory = "G:\Projects\Fluxonium\Data\Fluxonium #10 simulations\Input_for_auto_meas"
# simulation = "Trans_energy"
# path = directory + "\\" + simulation
# path = path + '_' + str(iState) + 'to' + str(fState) + '_from_' + str(current[0] * 1.0e3) + 'to' + str(
#     current[-1] * 1.0e3) + 'mA'
# current = current * 1e3
# print current
# sim_dat[:, 0] = current
# sim_dat[:, 1] = (energies[:, fState] - energies[:, iState])
# sim_dat[:, 2] = 10.3045
# np.savetxt(path + '.csv', sim_dat, delimiter=',', fmt='%.3f')
# plt.xlim([38.1,46.5])
# plt.ylim([0,5])
plt.tick_params(labelsize=18)

plt.show()
