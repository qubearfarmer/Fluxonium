# Analyze T1 data from 1.5-2.5 GHz
import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import charge_matrix_element as nem
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_cap as r_cap

plt.figure(figsize=(4, 4))
plt.rc('font', family='serif')

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

T1_array = []
T1_err_array = []
freq_array = []
flux_array = []
#########################################################################################
################################### T1 data, manual######################################
#########################################################################################
# 0-1 transition
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Summary of corrected flux"
simulation = "T1 avg_T2_qubit f(0to1) vs flux_all_new fit_012317_corrected flux.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',', dtype=float)
flux = data[1::, 0]
freq = data[1::, 1]
T1 = data[1::, 2]
T1_err = data[1::, 3]
T1_array = np.append(T1_array, T1)
T1_err_array = np.append(T1_err_array, T1_err)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)

#########################################################################################
################################### T1 data, automatic###################################
#########################################################################################
'''
simulation = "AUTO_T1_qubit f(0to1)_rabi vs flux 38p5to45mA_012317_corrected flux.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',', dtype=float)
flux = data[1::, 0]
freq = data[1::, 1]
T1 = data[1::, 2]
T1_err = data[1::, 3]
T1_array = np.append(T1_array, T1)
T1_err_array = np.append(T1_err_array, T1_err)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
'''
###################################Slice through the arrays###################################
# '''
T1_final = []
T1_err_final = []
flux_final = []
freq_final = []
for idx in range(len(T1_array)):
    if freq_array[idx] > 1.5 and freq_array[idx] < 2.5:
        T1_final = np.append(T1_final, T1_array[idx])
        T1_err_final = np.append(T1_err_final, T1_err_array[idx])
        flux_final = np.append(flux_final, flux_array[idx])
        freq_final = np.append(freq_final, freq_array[idx])
# '''
###############################################################################################
###################################Calculate matrix elements###################################
###############################################################################################
current = flux_final * 1e-3
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
    n_element[idx] = nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    p_element[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element[idx, :] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
#########################################################################################
################################### T1 data 02 transition################################
#########################################################################################
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Summary of corrected flux"
simulation = "T1 avg_T2_qubit f(0to2) vs flux_38p26 to 45mA_012517_corrected flux.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',', dtype=float)
flux = data[1::, 0]
freq = data[1::, 1]
T1 = data[1::, 2]
T1_err = data[1::, 3]

###################################Slice through the arrays###################################
# '''
current2 = flux * 1e-3
energies2 = np.zeros((len(current2), level_num))
iState = 1
fState = 2
for idx, curr in enumerate(current2):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies2[idx, idy] = H.eigenenergies()[idy]

T1_final2 = []
T1_err_final2 = []
flux_final2 = []
freq_final2 = []
for idx in range(len(T1)):
    if (energies2[idx, 2] - energies2[idx, 1]) > 1.5 and (energies2[idx, 2] - energies2[idx, 1]) < 2.5 \
            and (energies2[idx, 2] - energies2[idx, 0]) > 4.7:
        T1_final2 = np.append(T1_final2, T1[idx])
        T1_err_final2 = np.append(T1_err_final2, T1_err[idx])
        flux_final2 = np.append(flux_final2, flux[idx])
        freq_final2 = np.append(freq_final2, freq[idx])
    if (energies2[idx, 2] - energies2[idx, 1]) > 1.5 and (energies2[idx, 2] - energies2[idx, 1]) < 2.5 \
            and (energies2[idx, 2] - energies2[idx, 0]) > 4.1 and flux[idx] > 41 and flux[idx] < 42:
        T1_final2 = np.append(T1_final2, T1[idx])
        T1_err_final2 = np.append(T1_err_final2, T1_err[idx])
        flux_final2 = np.append(flux_final2, flux[idx])
        freq_final2 = np.append(freq_final2, freq[idx])
###############################################################################################
###################################Calculate matrix elements###################################
###############################################################################################
current2 = flux_final2 * 1e-3
qp_element21 = np.zeros((len(current2), 2))
n_element21 = np.zeros(len(current2))
p_element21 = np.zeros(len(current2))
for idx, curr in enumerate(current2):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    n_element21[idx] = nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    p_element21[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element21[idx,:] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
###############################################################################################
#######################################Simulation##############################################
###############################################################################################
pem = np.array([1e-2,2e0])
qpem = np.array([1e-6,10])
gamma_cap = np.zeros(len(pem))
gamma_qp = np.zeros((len(qpem),2))
w = 2

#Dielectric loss
# plt.errorbar(p_element ** 2, T1_final, yerr=T1_err_final, fmt='s', mfc='none', mew=2.0, mec='blue')
# plt.errorbar(p_element21 ** 2, T1_final2, yerr=T1_err_final2, fmt='d', mfc='none', mew=2.0, mec='red',ecolor = 'r')
# for Q_cap in [7e4, 7e5]:
#     for idx in range(len(pem)):
#         gamma_cap[idx] = r_cap(E_l, E_c, E_j_sum, d, Q_cap, w, pem[idx])
#     plt.loglog(pem**2, 1.0 / (gamma_cap) * 1e6, linewidth=2.0, linestyle ='--', color = 'k')
# fac = 7e3

#QP loss
plt.errorbar(qp_element[:, 0] ** 2 + qp_element[:, 1] ** 2, T1_final, yerr=T1_err_final, fmt='s', mfc='none', mew=2.0, mec='blue')
plt.errorbar(qp_element21[:, 0] ** 2 + qp_element21[:, 1] ** 2, T1_final2, yerr=T1_err_final2, fmt='d', mfc='none', mew=2.0, mec='red',ecolor = 'r')
for Q_qp in [10e5, 10e6]:
    for idx in range(len(qpem)):
        gamma_qp[idx,:] = r_qp(E_l, E_c, E_j_sum, d, Q_qp, w, [qpem[idx],qpem[idx]])
    plt.loglog((qpem)**2*2, 1.0 / (gamma_qp[:,0]+gamma_qp[:,1]) * 1e6, linewidth=2.0, linestyle ='--', color = 'k')
fac = 4e4
##########################################################################################
######################################Plots decoration###################################
##########################################################################################
ysmall = 1e1
ybig = 8e3
plt.ylim([ysmall, ybig])
plt.xlim([ysmall / fac, ybig / fac])
plt.tick_params(labelsize=18)
plt.xscale("log", nonposx='clip')
plt.yscale("log", nonposy='clip')

directory = 'C:\\Users\\nguyen89\\Box Sync\Research\Paper Images'
fname = 'T1_1.5to2.5GHz_pem.eps'
path = directory + '\\' + fname
plt.savefig(path, format='eps', dpi=1000)

plt.show()
