#T1 * matrix element^2 vs freq => spectral density
from matplotlib import pyplot as plt
import numpy as np
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import charge_matrix_element as nem
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Squid_small_junctions import relaxation_rate_cap as r_cap
from scipy.optimize import curve_fit

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
################################### T1 data 0 to 1 ######################################
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
T1_final1 = []
T1_err_final1 = []
flux_final1 = []
freq_final1 = []
Rabi_A_final1 = []
for idx in range(len(T1)):
    if flux[idx] == 38.604:
        continue
    if flux[idx] >= 38 and flux[idx] <= 47:
        T1_final1 = np.append(T1_final1, T1[idx])
        T1_err_final1 = np.append(T1_err_final1, T1_err[idx])
        flux_final1 = np.append(flux_final1, flux[idx])
        freq_final1 = np.append(freq_final1, freq[idx])
        Rabi_A_final1 = np.append(Rabi_A_final1, Rabi_A[idx])

#Calculate matrix elements
current1 = flux_final1 * 1e-3
qp_element1 = np.zeros((len(current1), 2))
p_element1 = np.zeros(len(current1))

iState = 0
fState = 1
for idx, curr in enumerate(current1):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    p_element1[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element1[idx, :] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
SPD_p1 = (T1_final1*1e-3*p_element1**2)**-1
SPD_qp1 = (T1_final1*1e-3*(qp_element1[:,0]**2+qp_element1[:,1]**2))**-1
#########################################################################################
################################### T1 data 0 to 2 ######################################
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
T1_final2 = []
T1_err_final2 = []
flux_final2 = []
freq_final2 = []
Rabi_A_final2 = []
#Only taking in points that are fluxons, hence having 2-1 decays dominating relaxation processes
for idx in range(len(T1)):
    if flux[idx] >= 38 and flux[idx] <= 47 and freq[idx] > 4.7:
        T1_final2 = np.append(T1_final2, T1[idx])
        T1_err_final2 = np.append(T1_err_final2, T1_err[idx])
        flux_final2 = np.append(flux_final2, flux[idx])
        freq_final2 = np.append(freq_final2, freq[idx])
        Rabi_A_final2 = np.append(Rabi_A_final2, Rabi_A[idx])
    if flux[idx] >= 41 and flux[idx] <= 42 and freq[idx] > 4.0:
        T1_final2 = np.append(T1_final2, T1[idx])
        T1_err_final2 = np.append(T1_err_final2, T1_err[idx])
        flux_final2 = np.append(flux_final2, flux[idx])
        freq_final2 = np.append(freq_final2, freq[idx])
        Rabi_A_final2 = np.append(Rabi_A_final2, Rabi_A[idx])

#Calculate matrix elements
current2 = flux_final2 * 1e-3
qp_element2 = np.zeros((len(current2), 2))
n_element2 = np.zeros(len(current2))
p_element2 = np.zeros(len(current2))
energy2 = np.zeros(len(current2))
iState = 1
fState = 2
for idx, curr in enumerate(current2):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    energy2[idx] = H.eigenenergies()[fState] - H.eigenenergies()[iState]
    p_element2[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element2[idx, :] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)

SPD_p2 = (T1_final2*1e-3*p_element2**2)**-1
SPD_qp2 = (T1_final2*1e-3*(qp_element2[:,0]**2+qp_element2[:,1]**2))**-1
#########################################################################################
####################################### Fitting ########################################
#########################################################################################
SPD_p = np.concatenate((SPD_p1, SPD_p2), axis=0)
SPD_qp = np.concatenate((SPD_qp1, SPD_qp2), axis=0)
freq = np.concatenate((freq_final1, energy2), axis=0)
freq_sim = np.linspace(0,50,501)

def SPD_func(freq, amp, x):
    return amp*freq**x

guess = ([SPD_p2[0], 1])
popt, pcov = curve_fit(SPD_func, freq, SPD_p)
SPD_pfit = SPD_func(freq_sim, popt[0], popt[1])
SPD_plim1 = SPD_func(freq_sim, popt[0], 0.9)
SPD_plim2 = SPD_func(freq_sim, popt[0], 2.5)
print popt[1]

guess = ([SPD_qp2[0], 1])
qpopt, qpcov = curve_fit(SPD_func, freq*1e9, SPD_qp)
SPD_qpfit = SPD_func(freq_sim, qpopt[0], qpopt[1])
SPD_qplim1 = SPD_func(freq_sim, qpopt[0], 0.5)
SPD_qplim2 = SPD_func(freq_sim, qpopt[0], 2.5)
print qpopt[1]
#######################################################################################################################
###########################################Plotting and decoration#####################################################
#######################################################################################################################
plt.figure(figsize=[10,10])

# plt.errorbar(freq_final1, SPD_p1 , fmt='s', mfc='none', mew=2.0, mec='b', ecolor ='b')
# plt.errorbar(energy2, SPD_p2, fmt='d', mfc='none', mew=2.0, mec='r', ecolor ='r')
# plt.plot(freq_sim, SPD_plim1, linewidth =2.0, color = 'black', linestyle ='--')
# plt.plot(freq_sim, SPD_plim2, linewidth =2.0, color = 'black', linestyle ='--')
# plt.plot(freq_sim, SPD_pfit, linewidth =2.0, color = 'black', linestyle ='--')

plt.errorbar(freq_final1, SPD_qp1, fmt='s', mfc='none', mew=2.0, mec='b', ecolor ='b')
plt.errorbar(energy2, SPD_qp2, fmt='d', mfc='none', mew=2.0, mec='r', ecolor ='r')
plt.plot(freq_sim, SPD_qplim1, linewidth =2.0, color = 'black', linestyle ='--')
plt.plot(freq_sim, SPD_qplim2, linewidth =2.0, color = 'black', linestyle ='--')
# # plt.plot(freq_sim, SPD_qpfit, linewidth =2.0, color = 'black', linestyle ='--')


plt.yscale("log")
plt.xscale("log")
xmin = 0.04
# xmax = 10
xmax = 3
fac = 3e2
plt.xlim([xmin,xmax])
plt.ylim((xmin*fac,xmax*fac))
plt.tick_params(labelsize = 24)

# directory = 'C:\\Users\\nguyen89\\Box Sync\Research\Paper Images'
# fname = 'SPD_pem.eps'
# path = directory + '\\' + fname
# plt.savefig(path, format='eps', dpi=1000)

plt.show()