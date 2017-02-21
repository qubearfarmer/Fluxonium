# Analyzing T1 data around 38p6mA for 0-1 transition
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
h = 6.62e-34  # Planck's constant
phi_o = h / (2 * e)  # Flux quantum

# Fitting for bottom spectrum
N = 50
E_l = 0.722729827116
E_c = 0.552669197076
E_j_sum = 17.61374383
A_j = 4.76321410213e-12
A_c = 1.50075181762e-10
d = 0.125005274368
beta_squid = 0.129912406349
beta_ext = 0.356925557542

# current = np.linspace(0.0384, 0.03875, 351)
current = np.linspace(0.0381, 0.0465, 841)
B_coeff = 60
level_num = 5

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
#########################################################################################
################################### T1 simulation ######################################
#########################################################################################
current = np.linspace(0.038,0.039,501)
qp_element = np.zeros((len(current), 2))
n_element = np.zeros(len(current))
p_element = np.zeros(len(current))
gamma_cap = np.zeros(len(current))
gamma_qp = np.zeros((len(current), 2))
energies = np.zeros((len(current), level_num))
iState = 0
fState = 1
for idx, curr in enumerate(current):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx, idy] = H.eigenenergies()[idy]
    p_element[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element[idx, :] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
w = energies[:, 1] - energies[:, 0]
fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax = plt.gca()
ax.set_yscale('log')
ax1.errorbar(flux_final, T1_final, yerr=T1_err_final, fmt='s', mfc='none', mew=2.0, mec='blue')

#Dielectric loss
for Q_cap in [5e4, 5e5]:
    for idx in range(len(current)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j_sum, d, Q_cap, w[idx], p_element[idx])
    ax1.plot(current*1e3, 1.0 / (gamma_cap) * 1e6, linewidth=2.0, linestyle ='--', color = 'k')

#QP loss
for Q_qp in [2e5, 2e6]:
    for idx in range(len(current)):
        gamma_qp[idx,:] = r_qp(E_l, E_c, E_j_sum, d, Q_qp, w[idx], qp_element[idx,:])
    ax1.plot(current*1e3, 1.0 / (gamma_qp[:,0]+gamma_qp[:,1]) * 1e6, linewidth=2.0, linestyle ='-.', color = 'k')

ax2 = ax1.twinx()
# ax.set_yscale('linear')
ax2.plot(current*1e3, w, linewidth = 2.0, color = 'blue')
##########################################################################################
######################################Plots decoration###################################
##########################################################################################
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
xmin = 38.5
xmax = 38.7
y2min = 3.2
y2max = 4.8
y1min = 6
y1max =3500
ax1.set_ylim([y1min, y1max])
ax1.set_xlim([xmin, xmax])
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([y2min, y2max])
ax1.set_xticks([38.55, 38.6, 38.65])
ax2.set_yticks(np.linspace(3.3,4.7, 3))
ax.get_xaxis().get_major_formatter().set_useOffset(False)

plt.show()

