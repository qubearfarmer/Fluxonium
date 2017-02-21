# All T1 data with spectrum simulation
from matplotlib import pyplot as plt
import numpy as np
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import charge_matrix_element as nem
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_cap as r_cap

plt.rc('font', family='serif')

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

fig, ax1 = plt.subplots(figsize=(20, 10))

#########################################################################################
################################### T1 data, manual######################################
#########################################################################################
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Summary of corrected flux"
simulation = "T1 avg_T2_qubit f(0to1) vs flux_all_new fit_012317_corrected flux.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',', dtype=float)
flux = data[1::, 0]
freq = data[1::, 1]
T1 = data[1::, 2]
T1_err = data[1::, 3]
Rabi_A = data[1::, 6]

###################################Slice through the arrays###################################

T1_final = []
T1_err_final = []
flux_final = []
freq_final = []
Rabi_A_final = []
for idx in range(len(T1)):
    if flux[idx] == 38.604:
        continue
    if flux[idx] >= 38 and flux[idx] <= 47:#38.7:
        T1_final = np.append(T1_final, T1[idx])
        T1_err_final = np.append(T1_err_final, T1_err[idx])
        flux_final = np.append(flux_final, flux[idx])
        freq_final = np.append(freq_final, freq[idx])
        Rabi_A_final = np.append(Rabi_A_final, Rabi_A[idx])

ax1.errorbar(flux_final, T1_final, yerr=T1_err_final, fmt='s', mfc='none', mew=2.0, mec='blue')
# ax1.errorbar(flux_final, freq_final,  fmt='s', mfc='none', mew=2.0, mec='blue')
###############################################################################################
#02 data
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Summary of corrected flux"
simulation = "T1 avg_T2_qubit f(0to2) vs flux_38p26 to 45mA_012517_corrected flux.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',', dtype=float)
flux = data[1::, 0]
freq = data[1::, 1]
T1 = data[1::, 2]
T1_err = data[1::, 3]
Rabi_A = data[1::, 5]
T1_final = []
T1_err_final = []
flux_final = []
freq_final = []
Rabi_A_final = []
for idx in range(len(T1)):
    if flux[idx] >= 38 and flux[idx] <= 47 and freq[idx] > 4.7:
        T1_final = np.append(T1_final, T1[idx])
        T1_err_final = np.append(T1_err_final, T1_err[idx])
        flux_final = np.append(flux_final, flux[idx])
        freq_final = np.append(freq_final, freq[idx])
        Rabi_A_final = np.append(Rabi_A_final, Rabi_A[idx])
    if flux[idx] >= 41 and flux[idx] <= 42 and freq[idx] > 4.0:
        T1_final = np.append(T1_final, T1[idx])
        T1_err_final = np.append(T1_err_final, T1_err[idx])
        flux_final = np.append(flux_final, flux[idx])
        freq_final = np.append(freq_final, freq[idx])
        Rabi_A_final = np.append(Rabi_A_final, Rabi_A[idx])

ax1.errorbar(flux_final, T1_final, yerr=T1_err_final, fmt='d', mfc='none', mew=2.0, mec='r', ecolor ='r')
# ax1.errorbar(flux_final, freq_final, fmt='d', mfc='none', mew=2.0, mec='r', ecolor ='r')
###############################################################################################
xmin = 38.1
xmax = 45.5
y1min = 6
y1max =3500
ax1.set_ylim([y1min, y1max])
ax1.set_xlim([xmin, xmax])
ax = plt.gca()
ax.set_yscale('log')
plt.tick_params(labelsize = 18)
#########################################################################################
################################### Simulation spectrum #################################
#########################################################################################
ax2 = ax1.twinx()
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
current = np.linspace(0.038, 0.047, 901)
energies = np.zeros((len(current),level_num))
#Compute eigenenergies
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

ax2.plot(current*1e3, energies[:,1]-energies[:,0], linewidth = 1.0, color = 'blue')
ax2.plot(current*1e3, energies[:,2]-energies[:,0], linewidth = 1.0, color = 'red')
###############################################################################################
xmin = 38.1
xmax = 45.5
y2min = 2
y2max =8
ax2.set_ylim([y2min, y2max])
ax2.set_xlim([xmin, xmax])
ax = plt.gca()
plt.tick_params(labelsize = 18)

directory = 'C:\\Users\\nguyen89\\Box Sync\Research\Paper Images'
fname = 'T1_all.eps'
path = directory + '\\' + fname
fig.savefig(path, format='eps', dpi=1000)

plt.show()
