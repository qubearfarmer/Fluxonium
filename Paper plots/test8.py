# Analyzing Rabi amplitude around 38p6mA for 0-2 transition
import numpy as np
from matplotlib import pyplot as plt

plt.rc('font', family='serif')
# Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Dispersive_shifts_wSquid"
path = directory + "\\" + simulation
# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

# Qubit and computation parameters
N = 50
E_l = 0.722729827116
E_c = 0.552669197076
E_j_sum = 17.61374383
A_j = 4.76321410213e-12
A_c = 1.50075181762e-10
d = 0.125005274368
beta_squid = 0.129912406349
beta_ext = 0.356925557542

current = np.linspace(0.038, 0.039, 1001)
chi = np.zeros(len(current))
level_num = 5
energies = np.zeros((len(current), level_num))

iState = 0
fState = 2
B_coeff = 60
wr = 10.304
g = 0.084
path = path + "_" + str(iState) + str(fState) + "_" + str(current[0] * 1e3) + "to" + str(current[-1] * 1e3) + "mA"
#######################################################################################################################
# Simulation part
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
# Plotting part
path = directory + "\\" + simulation
path = path + "_" + str(iState) + str(fState) + "_" + str(current[0] * 1e3) + "to" + str(current[-1] * 1e3) + "mA"
energies = np.genfromtxt(path + "_energies.txt")
chi = np.genfromtxt(path + "_chi.txt")

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.set_xticks([38.3, 38.4, 38.5, 38.6, 38.7])
ax1.tick_params(labelsize=18)
ax1.plot(current * 1e3, abs(chi) * 1e3, linestyle='--', color='k', linewidth='2')
############################################################################################################
iState = 0
fState = 1
path = directory + "\\" + simulation
path = path + "_" + str(iState) + str(fState) + "_" + str(current[0] * 1e3) + "to" + str(current[-1] * 1e3) + "mA"
energies = np.genfromtxt(path + "_energies.txt")
chi = np.genfromtxt(path + "_chi.txt")
ax1.plot(current * 1e3, abs(chi) * 1e3, linestyle='-.', color='k', linewidth='2')

ax1.set_xlim([38.2, 38.8])
ax1.set_ylim([0.1, 0.5])
################################################################################################################
##########################################Rabi data####################################################
################################################################################################################
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Summary of corrected flux"
simulation = "T1 avg_T2_qubit f(0to2) vs flux_38p26 to 45mA_012517_corrected flux.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',', dtype=float)
flux = data[1::, 0]
freq = data[1::, 1]
T1 = data[1::, 2]
T1_err = data[1::, 3]
Rabi_A = data[1::, 5]

###################################Slice through the arrays###################################
# '''
T1_final = []
T1_err_final = []
flux_final = []
freq_final = []
Rabi_A_final = []
for idx in range(len(T1)):
    if Rabi_A[idx] == 'NaN':
        continue
    if flux[idx] > 38 and flux[idx] < 39:
        T1_final = np.append(T1_final, T1[idx])
        T1_err_final = np.append(T1_err_final, T1_err[idx])
        flux_final = np.append(flux_final, flux[idx])
        freq_final = np.append(freq_final, freq[idx])
        Rabi_A_final = np.append(Rabi_A_final, Rabi_A[idx])

ax2 = ax1.twinx()
ax2.errorbar(flux_final, Rabi_A_final, fmt='d', mfc='none', mew='2', mec='red', ecolor='red')
ax2.set_xlim([38.2, 38.8])
# ax2.set_ylim([0,4])
ax2.tick_params(labelsize=18)
plt.show()
