# Analyzing T1 data around 38p6mA
import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem

plt.figure(figsize=(10, 10))
plt.rc('font', family='serif')
# Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Relaxation_wSquid"
path = directory + "\\" + simulation

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

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
# '''
T1_final = []
T1_err_final = []
flux_final = []
freq_final = []
Rabi_A_final = []
for idx in range(len(T1)):
    if flux[idx] >= 38.523 and flux[idx] <= 38.75:
        T1_final = np.append(T1_final, T1[idx])
        T1_err_final = np.append(T1_err_final, T1_err[idx])
        flux_final = np.append(flux_final, flux[idx])
        freq_final = np.append(freq_final, freq[idx])
        Rabi_A_final = np.append(Rabi_A_final, Rabi_A[idx])
# '''

# plt.plot(flux_final, Rabi_A_final, 'ro')

ax = plt.gca()
ax.set_yscale('linear')
plt.errorbar(flux_final, T1_final, yerr=T1_err_final, fmt='s', mfc='none', mew='2', mec='blue')
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.xlim([38.55, 38.75])
# plt.ylim([4.2,5.1])
plt.xticks([38.55, 38.6, 38.65, 38.7, 38.75])
# plt.yticks([4.2, 4.6, 5.0])
plt.tick_params(labelsize=18)

#########################################################################################
################################### T1 simulation ######################################
#########################################################################################
current = np.linspace(38.55, 38.75, 21) * 1e-3
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
energies = np.zeros((len(current), level_num))
qp_element = np.zeros((len(current), 2))
n_element = np.zeros(len(current))
p_element = np.zeros(len(current))

iState = 0
fState = 1
for idx, curr in enumerate(current):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx, idy] = H.eigenenergies()[idy]
    # n_element[idx] = nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
    #                      2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    p_element[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element[idx, :] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)

hbar = h / (2 * np.pi)
kB = 1.38064852e-23
T = 1e-2
E_c = E_c / 1.509190311677e+24  # convert GHz to J
E_l = E_l / 1.509190311677e+24  # convert to J
E_j_sum = E_j_sum / 1.509190311677e+24  # convert to J
E_j1 = 0.5 * E_j_sum * (1 + d)
E_j2 = 0.5 * E_j_sum * (1 - d)
delta_alum = 5.447400321e-23  # J

########################################Upper limit########################################
Q_cap = 0.55e6
Q_ind = 0.8e6
Q_qp = 2.7e6

cap = e ** 2 / (2.0 * E_c)
ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
gk = e ** 2.0 / h
g1 = 8.0 * E_j1 * gk / delta_alum
g2 = 8.0 * E_j2 * gk / delta_alum

trans_energy = energies[:, fState] - energies[:, iState]
# w = trans_energy*1e9*2*np.pi
w = 4e9 * 2 * np.pi
Y_cap = w * cap / Q_cap
Y_ind = 1.0 / (w * ind * Q_ind)
Y_qp1 = (g1 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)
Y_qp2 = (g2 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)

gamma_cap = np.zeros(len(current))
gamma_qp = np.zeros((len(current), 2))

for idx in range(len(current)):
    gamma_cap[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_cap * (
        1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    # gamma_ind[idx] = (phi_o * pem_sim[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_ind * (1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    gamma_qp[idx, 0] = (qp_element[idx, 0]) ** 2 * (w / np.pi / gk) * Y_qp1
    gamma_qp[idx, 1] = (qp_element[idx, 1]) ** 2 * (w / np.pi / gk) * Y_qp2
T1_sim = 1 / gamma_cap
plt.plot(current * 1e3, T1_sim * 1e6, linewidth='2', color='k', linestyle='--')
# T1_sim = 1 / (gamma_qp[:, 0] + gamma_qp[:, 1])
# plt.loglog(2 * qpem_sim ** 2, T1_sim * 1e6, linewidth='2', color='k', linestyle='--')

########################################Lower limit########################################
Q_cap = 0.047e6
Q_ind = 0.8e6
Q_qp = 0.177e6

cap = e ** 2 / (2.0 * E_c)
ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
gk = e ** 2.0 / h
g1 = 8.0 * E_j1 * gk / delta_alum
g2 = 8.0 * E_j2 * gk / delta_alum
trans_energy = energies[:, fState] - energies[:, iState]
# w = trans_energy*1e9*2*np.pi
w = 4e9 * 2 * np.pi
Y_cap = w * cap / Q_cap
Y_ind = 1.0 / (w * ind * Q_ind)
Y_qp1 = (g1 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)
Y_qp2 = (g2 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)

gamma_cap = np.zeros(len(current))
gamma_qp = np.zeros((len(current), 2))

for idx in range(len(current)):
    gamma_cap[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_cap * (
        1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    # gamma_ind[idx] = (phi_o * pem_sim[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_ind * (1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    gamma_qp[idx, 0] = (qp_element[idx, 0]) ** 2 * (w / np.pi / gk) * Y_qp1
    gamma_qp[idx, 1] = (qp_element[idx, 1]) ** 2 * (w / np.pi / gk) * Y_qp2
T1_sim = 1 / gamma_cap
plt.plot(current * 1e3, T1_sim * 1e6, linewidth='2', color='k', linestyle='--')
# T1_sim = 1 / (gamma_qp[:, 0] + gamma_qp[:, 1])
# plt.loglog(2 * qpem_sim ** 2, T1_sim * 1e6, linewidth='2', color='k', linestyle='--')
plt.show()
