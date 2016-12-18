# Analyzing T1 data around 38p6mA
import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import charge_matrix_element as nem
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

# T1 data for 01 transition
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Corrected flux"
T1_array = []
flux_array = []
freq_array = []
simulation = "T1 avg_T2_qubit f(0to1) vs flux_38p5 to 38p76mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',')
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
# rabiA = data[:,5]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)

directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Automation code\corrected flux with Rabi A new"

simulation = "T1_rabi_38p5to38p6mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)

simulation = "T1_rabi_38p58to38p62mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)

simulation = "T1_rabi_38p62to38p68mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)

# Slice through the arrays


# Get matrix elements
current = flux_array * 1.0e-3
N = 50
E_l = 0.746959655208
E_c = 0.547943694372
E_j_sum = 21.9627179709
level_num = 10
B_coeff = 60
A_j = 3.80888914574e-12
A_c = 1.49982268962e-10
beta_squid = 0.00378012644185
beta_ext = 0.341308382441
d = 0.0996032153487
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
trans_energy = energies[:, fState] - energies[:, iState]
w = trans_energy * 1e9 * 2 * np.pi

# plt.loglog(p_element**2, T1_array*trans_energy**2, 's', mfc='none', mew='2', mec='blue')
plt.loglog(qp_element[:, 0] ** 2 + qp_element[:, 1] ** 2, T1_array / np.sqrt(trans_energy), 's', mfc='none', mew='2',
           mec='blue')

############Simulation

hbar = h / (2 * np.pi)
kB = 1.38064852e-23
T = 1e-2
E_c = E_c / 1.509190311677e+24  # convert GHz to J
E_l = E_l / 1.509190311677e+24  # convert to J
E_j_sum = E_j_sum / 1.509190311677e+24  # convert to J
E_j1 = 0.5 * E_j_sum * (1 + d)
E_j2 = 0.5 * E_j_sum * (1 - d)
delta_alum = 5.447400321e-23  # J

Q_cap = 0.3e6
Q_ind = 0.8e6
Q_qp = 2e6

cap = e ** 2 / (2.0 * E_c)
ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
gk = e ** 2.0 / h
g1 = 8.0 * E_j1 * gk / delta_alum
g2 = 8.0 * E_j2 * gk / delta_alum

trans_energy = energies[:, fState] - energies[:, iState]
w = trans_energy * 1e9 * 2 * np.pi
# plt.plot(current*1e3+0.016, trans_energy,'b--')
# plt.plot(current*1e3+0.016, energies[:,2]-energies[:,0],'b--')
Y_cap = w * cap / Q_cap
Y_ind = 1.0 / (w * ind * Q_ind)
Y_qp1 = (g1 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)
Y_qp2 = (g2 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)

gamma_cap = np.zeros(len(current))
gamma_ind = np.zeros(len(current))
gamma_qp = np.zeros((len(current), 2))

for idx in range(len(current)):
    gamma_cap[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_cap[idx] * (
    1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    gamma_ind[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_ind[idx] * (
    1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    gamma_qp[idx, 0] = (qp_element[idx, 0]) ** 2 * (w[idx] / np.pi / gk) * Y_qp1[idx]
    gamma_qp[idx, 1] = (qp_element[idx, 1]) ** 2 * (w[idx] / np.pi / gk) * Y_qp2[idx]

# gamma_cap = gamma_cap / (trans_energy)**2
# pem_final = np.array([np.min(p_element),np.max(p_element)])
# gam_final = np.array([np.min(gamma_cap),np.max(gamma_cap)])
# plt.loglog(pem_final**2, 1/gam_final * 1e6, 'k--', linewidth = '2')


gamma_qp = (gamma_qp[:, 0] + gamma_qp[:, 1]) * np.sqrt(trans_energy)
qpem_final = np.array(
    [np.min(qp_element[:, 0] ** 2 + qp_element[:, 1] ** 2), np.max(qp_element[:, 0] ** 2 + qp_element[:, 1] ** 2)])
gam_final = np.array([np.min(gamma_qp), np.max(gamma_qp)])
plt.loglog(qpem_final, 1 / gam_final * 1e6, 'k--', linewidth='2')
#######################################Plotting stuff
# fac = 1e6
# plt.ylim([2e2,6e4])
# plt.xlim([2e2/fac,6e4/fac])

fac = 5e4
plt.ylim([4, 2e3])
plt.xlim([4 / fac, 2e3 / fac])
plt.tick_params(labelsize=18)
plt.show()
