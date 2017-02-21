# Analyzing T1 data around 38p6mA for 0-2 transition
from matplotlib import pyplot as plt
import numpy as np
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_cap as r_cap

plt.rc('font', family='serif')

# Define constants and parameters
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

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
#########################################################################################
################################### T1 data, manual######################################
#########################################################################################
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

T1_final = []
T1_err_final = []
flux_final = []
freq_final = []
for idx in range(len(T1)):
    if Rabi_A[idx] > 0 and flux[idx] > 38.2 and flux[idx] < 38.8:
        T1_final = np.append(T1_final, T1[idx])
        T1_err_final = np.append(T1_err_final, T1_err[idx])
        flux_final = np.append(flux_final, flux[idx])
        freq_final = np.append(freq_final, freq[idx])

#########################################################################################
################################### T1 simulation ######################################
#########################################################################################
current = np.linspace(0.0382,0.0388,101)
energies = np.zeros((len(current), level_num))

qp_element = np.zeros((len(current), 2))
n_element = np.zeros(len(current))
p_element = np.zeros(len(current))
gamma_cap_up = np.zeros(len(current))
gamma_cap_low = np.zeros(len(current))
gamma_qp_up = np.zeros((len(current), 2))
gamma_qp_low = np.zeros((len(current), 2))

qp_element21 = np.zeros((len(current), 2))
n_element21 = np.zeros(len(current))
p_element21 = np.zeros(len(current))
gamma_cap_up21 = np.zeros(len(current))
gamma_cap_low21 = np.zeros(len(current))
gamma_qp_up21 = np.zeros((len(current), 2))
gamma_qp_low21 = np.zeros((len(current), 2))

####################################################################################
iState = 0
fState = 2
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
w = energies[:, fState] - energies[:, iState]

Q_cap = 5e4
for idx in range(len(current)):
    gamma_cap_low[idx] = r_cap(E_l, E_c, E_j_sum, d, Q_cap, w[idx], p_element[idx])
Q_cap = 5e5
for idx in range(len(current)):
    gamma_cap_up[idx] = r_cap(E_l, E_c, E_j_sum, d, Q_cap, w[idx], p_element[idx])
Q_qp = 2e5
for idx in range(len(current)):
    gamma_qp_low[idx,:] = r_qp(E_l, E_c, E_j_sum, d, Q_qp, w[idx], qp_element[idx,:])
Q_qp = 2e6
for idx in range(len(current)):
    gamma_qp_up[idx,:] = r_qp(E_l, E_c, E_j_sum, d, Q_qp, w[idx], qp_element[idx,:])

#########################################################################################
iState = 1
fState = 2
for idx, curr in enumerate(current):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    p_element21[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element21[idx, :] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
w21 = energies[:, fState] - energies[:, iState]

Q_cap = 5e4
for idx in range(len(current)):
    gamma_cap_low21[idx] = r_cap(E_l, E_c, E_j_sum, d, Q_cap, w21[idx], p_element21[idx])
Q_cap = 5e5
for idx in range(len(current)):
    gamma_cap_up21[idx] = r_cap(E_l, E_c, E_j_sum, d, Q_cap, w21[idx], p_element21[idx])
Q_qp = 5e5
for idx in range(len(current)):
    gamma_qp_low21[idx,:] = r_qp(E_l, E_c, E_j_sum, d, Q_qp, w21[idx], qp_element21[idx,:])
Q_qp = 25e6
for idx in range(len(current)):
    gamma_qp_up21[idx,:] = r_qp(E_l, E_c, E_j_sum, d, Q_qp, w21[idx], qp_element21[idx,:])

######################################################################################################
fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax = plt.gca()
ax.set_yscale('log')
ax1.errorbar(flux_final, T1_final, yerr=T1_err_final, fmt='d', mfc='none', mew=2.0, mec='red', ecolor='red')
ax1.plot(current*1e3, 1.0 / (gamma_cap_up21 + gamma_cap_up) * 1e6, linewidth=2.0, linestyle ='--', color = 'k')
ax1.plot(current*1e3, 1.0 / (gamma_cap_low21 + gamma_cap_low) * 1e6, linewidth=2.0, linestyle ='--', color = 'k')
ax1.plot(current*1e3, 1.0 / (gamma_qp_up21[:,0] + gamma_qp_up21[:,1]+gamma_qp_up[:,0] + gamma_qp_up[:,1]) * 1e6, linewidth=2.0, linestyle ='-.', color = 'k')
ax1.plot(current*1e3, 1.0 / (gamma_qp_low21[:,0] + gamma_qp_low21[:,1]+gamma_qp_low[:,0] + gamma_qp_low[:,1]) * 1e6, linewidth=2.0, linestyle ='-.', color = 'k')
ax2 = ax1.twinx()
ax2.plot(current*1e3, w21, linewidth = 2.0, color = 'red')
##########################################################################################
######################################Plots decoration###################################
##########################################################################################
ymin2 = 0
ymax2 = 3.7
xmin = 38.2
xmax = 38.56
ymin1 = 5
ymax1 = 1000
ax1.set_ylim([ymin1, ymax1])
ax1.set_xlim([xmin, xmax])
ax1.set_xticks([38.2, 38.3, 38.4, 38.5])
ax2.set_yticks([0, 1, 2, 3])
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin2, ymax2])
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)

plt.show()