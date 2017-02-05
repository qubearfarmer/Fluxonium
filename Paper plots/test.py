from matplotlib import pyplot as plt
import numpy as np
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem
#Define constants
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

T1_array = []
T1_err_array = []
freq_array = []
flux_array = []

directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Summary of corrected flux"
simulation = "T1 avg_T2_qubit f(0to2) vs flux_38p26 to 45mA_012517_corrected flux.csv"
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
# plt.plot(flux_array, freq_array, 'ro')

###################################Slice through the arrays###################################
# '''
current = flux_array * 1e-3
# current = np.linspace(0.038,0.046, 801)
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

iState = 1
fState = 2
for idx, curr in enumerate(current):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx, idy] = H.eigenenergies()[idy]

T1_final = []
T1_err_final = []
flux_final = []
freq_final = []

# plt.plot(current*1e3, energies[:,2]-energies[:,0])
for idx in range(len(T1_array)):
    # if T1_array[idx] < 15:
    #     continue
    if (energies[idx,2] - energies[idx,1]) > 1.5 and (energies[idx,2] - energies[idx,1]) < 2.5 \
        and (energies[idx,2] - energies[idx,0]) > 5:
        T1_final = np.append(T1_final, T1_array[idx])
        T1_err_final = np.append(T1_err_final, T1_err_array[idx])
        flux_final = np.append(flux_final, flux_array[idx])
        freq_final = np.append(freq_final, freq_array[idx])
plt.plot(flux_final, freq_final, 'r.')

current = np.linspace(0.037,0.047,101)
energies = np.zeros((len(current), level_num))
for idx, curr in enumerate(current):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx, idy] = H.eigenenergies()[idy]
plt.plot(current*1e3, energies[:, 1] - energies[:, 0])
plt.plot(current*1e3, energies[:, 2] - energies[:, 0])
plt.show()
