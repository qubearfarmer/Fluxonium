import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap

# plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.figure(figsize=[7, 4])
root = "C:\\Users\\nguyen89"

e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum
kB = 1.38e-23
T_diel = 20.0e-3
alpha = 0.15
Q1 = 0.28e6
Q2 = 0.5e6
fState = 1
iState = 0

###################################Fluxonium #13###################################
directory = "Box\Python Codes\Fluxonium simulation results"
fname = "Relaxation_13"
path = root + "\\" + directory + "\\" + fname
E_j = 3
E_l = 1
E_c = 0.84
energies = np.genfromtxt(path + '_energies.txt')
n_element = np.genfromtxt(path + '_chargeElement.txt')
p_element = np.genfromtxt(path + '_fluxElement.txt')
qp_element = np.genfromtxt(path + '_qpElement.txt')
w = energies[:, fState] - energies[:, iState]
gamma_cap = np.zeros(len(p_element))
matrix_element_interp = np.genfromtxt(path + '_matrix_element_interp.txt')
thermal_factor_diel = (1 + np.exp(-h * w * 1e9 / (kB * T_diel)))

# for Q_cap in [Q1, Q2]:
#     for idx in range(len(p_element)):
#         gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap*(5.0/w[idx])**alpha, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
#     plt.loglog(w, 1.0 / gamma_cap*1e6 * p_element ** 2, linewidth=2.0, linestyle='--')

directory = 'Box\Research\Paper Images\Coherence summary\\Fluxonium #13'
fname = 'T1_60to62mA.txt'
path = root + "\\" + directory + "\\" + fname
data = np.genfromtxt(path, skip_header=1)  # , delimiter= ',')
current = data[:, 0]
freq = data[:, 1]
T1 = data[:, 2]
plt.errorbar(freq, T1 * matrix_element_interp ** 2, fmt='s', mfc='none', mew=2.0, mec='b', label="Qubit A")

###################################Fluxonium #28###################################
directory = "Box\Python Codes\Fluxonium simulation results"
fname = "Relaxation_28"
path = root + "\\" + directory + "\\" + fname
E_j = 4.86
E_l = 1.14
E_c = 0.84
energies = np.genfromtxt(path + '_energies.txt')
n_element = np.genfromtxt(path + '_chargeElement.txt')
p_element = np.genfromtxt(path + '_fluxElement.txt')
qp_element = np.genfromtxt(path + '_qpElement.txt')
w = energies[:, fState] - energies[:, iState]
gamma_cap = np.zeros(len(p_element))
matrix_element_interp = np.genfromtxt(path + '_matrix_element_interp.txt')
thermal_factor_diel = (1 + np.exp(-h * w * 1e9 / (kB * T_diel)))

for Q_cap in [Q1, Q2]:
    for idx in range(len(p_element)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap * (6.0 / w[idx]) ** alpha, w[idx], p_element[idx], T_diel) * \
                         thermal_factor_diel[idx]
    plt.loglog(w, 1.0 / gamma_cap * 1e6 * p_element ** 2, linewidth=2.0, linestyle='--', color='orange')

'''
for Q_cap in [Q1, Q2]:
    for idx in range(len(p_element)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
    plt.loglog(w, 1.0 / gamma_cap*1e6 * p_element ** 2, linewidth=2.0, linestyle='-', color = 'orange')
'''
directory = 'Box\Research\Paper Images\Coherence summary\\Fluxonium #28'
fname = 'T1_summary_2018_04_13.txt'
path = root + "\\" + directory + "\\" + fname
data = np.genfromtxt(path)
current = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
T1_err = data[1:, 3]
plt.errorbar(freq, T1 * matrix_element_interp ** 2, fmt='s', mfc='none', mew=2.0, mec='g', ecolor='g', label='Qubit B')

###################################Fluxonium #32###################################
directory = "Box\Python Codes\Fluxonium simulation results"
fname = "Relaxation_32"
path = root + "\\" + directory + "\\" + fname
E_j = 1.65
E_l = 0.19
E_c = 1.14
energies = np.genfromtxt(path + '_energies.txt')
n_element = np.genfromtxt(path + '_chargeElement.txt')
p_element = np.genfromtxt(path + '_fluxElement.txt')
qp_element = np.genfromtxt(path + '_qpElement.txt')
w = energies[:, fState] - energies[:, iState]
gamma_cap = np.zeros(len(p_element))
matrix_element_interp = np.genfromtxt(path + '_matrix_element_interp.txt')
thermal_factor_diel = (1 + np.exp(-h * w * 1e9 / (kB * T_diel)))

# for Q_cap in [Q1, Q2]:
#     for idx in range(len(p_element)):
#         gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap*(5.0/w[idx])**alpha, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
#     plt.loglog(w, 1.0 / gamma_cap*1e6 * p_element ** 2*0.84/1.14, linewidth=2.0, linestyle='--')

directory = 'Box\Research\Paper Images\Coherence summary\\Fluxonium #32'
fname = 'T1 summary 2018_08_28.txt'
path = root + "\\" + directory + "\\" + fname
data = np.genfromtxt(path)
current = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
T1_err = data[1:, 3]
plt.errorbar(freq, T1 * matrix_element_interp ** 2 * 0.84 / 1.14, fmt='s', mfc='none', mew=2.0, mec='red', ecolor='red',
             label='Qubit G')

###################################Fluxonium #33###################################
directory = "Box\Python Codes\Fluxonium simulation results"
fname = "Relaxation_33"
path = root + "\\" + directory + "\\" + fname
E_j = 4.43
E_l = 0.79
E_c = 0.985
energies = np.genfromtxt(path + '_energies.txt')
n_element = np.genfromtxt(path + '_chargeElement.txt')
p_element = np.genfromtxt(path + '_fluxElement.txt')
qp_element = np.genfromtxt(path + '_qpElement.txt')
w = energies[:, fState] - energies[:, iState]
gamma_cap = np.zeros(len(p_element))
matrix_element_interp = np.genfromtxt(path + '_matrix_element_interp.txt')
thermal_factor_diel = (1 + np.exp(-h * w * 1e9 / (kB * T_diel)))

# for Q_cap in [Q1, Q2]:
#     for idx in range(len(p_element)):
#         gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap*(6.0/w[idx])**alpha, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
#     plt.loglog(w, 1.0 / gamma_cap*1e6 * p_element ** 2*0.84/0.985, linewidth=2.0, linestyle='--', color = 'orange')

directory = 'Box\Research\Paper Images\Coherence summary\\Fluxonium #33'
fname = 'T1 summary 2018_09_28.txt'
path = root + "\\" + directory + "\\" + fname
data = np.genfromtxt(path)
current = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
T1_err = data[1:, 3]
plt.errorbar(freq, T1 * matrix_element_interp ** 2 * 0.84 / 0.985, fmt='s', mfc='none', mew=2.0, mec='k', ecolor='k',
             label='Qubit H')

plt.tick_params(labelsize=18.0)
plt.legend()
plt.xticks([])
plt.yticks([])
plt.show()
