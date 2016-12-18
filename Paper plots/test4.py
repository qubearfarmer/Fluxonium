# Analyze T1 data vs Rabi amplitude
import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 10))
plt.rc('font', family='serif')

T1_array = []
freq_array = []
flux_array = []
rabiA_array = []
# Define file directory
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Corrected flux\With Rabi A"
simulation = "T1 avg_T2_qubit f(0to1) vs flux_38p5 to 38p76mA_with Rabi A.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',')
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 5]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)
# plt.semilogy(rabiA, T1,  's', mfc='none', mew='2', mec='blue')

simulation = "T1 avg_T2_qubit f(0to1) vs flux_39p46 to 39p39mA_with Rabi A.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',')
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 5]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)
# plt.semilogy(rabiA, T1,  's', mfc='none', mew='2', mec='red')

simulation = "T1_T2_qubit f(0to1)vs flux 43p65_45p4mA_with Rabi A.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',')
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 5]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)
# plt.semilogy(rabiA, T1,  's', mfc='none', mew='2', mec='green')

simulation = "T1avg(0to1)vs flux 41p52 to 42p0mA_with Rabi A.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter=',')
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 4]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)
# plt.semilogy(rabiA, T1,  's', mfc='none', mew='2', mec='yellow')

directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Automation code\corrected flux with Rabi A new"
simulation = "T1_rabi_35to36p1mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 3]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)

simulation = "T1_rabi_37p2to38p5mA_5usStep_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 3]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)

simulation = "T1_rabi_38p5to38p6mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 3]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)

simulation = "T1_rabi_38p58to38p62mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 3]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)

simulation = "T1_rabi_38p62to38p68mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 3]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)

simulation = "T1_rabi_41p55to41p6mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 3]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)

simulation = "T1_rabi_41to41p05mA_corrected flux.TXT"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
flux = data[1:, 0]
freq = data[1:, 1]
T1 = data[1:, 2]
rabiA = data[1:, 3]
T1_array = np.append(T1_array, T1)
flux_array = np.append(flux_array, flux)
freq_array = np.append(freq_array, freq)
rabiA_array = np.append(rabiA_array, rabiA)

#####################################Slice through array#########################################
T1_final = []
flux_final = []
freq_final = []
rabiA_final = []
for idx in range(len(T1_array)):
    if rabiA_array[idx] > 0:
        T1_final = np.append(T1_final, T1_array[idx])
        flux_final = np.append(flux_final, flux_array[idx])
        freq_final = np.append(freq_final, freq_array[idx])
        rabiA_final = np.append(rabiA_final, rabiA_array[idx])
'''
#################################################################################################
current = flux_final*1e-3
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
d=0.0996032153487
energies = np.zeros((len(current),level_num))
qp_element = np.zeros((len(current),2))
n_element = np.zeros(len(current))
p_element = np.zeros(len(current))

iState = 0
fState = 1
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx, idy] = H.eigenenergies()[idy]
    n_element [idx] = nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    p_element[idx] = pem(  N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element[idx,:] = qpem(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)

# plt.loglog(qp_element[:,0]**2 + qp_element[:,1]**2, T1_final, 's', mfc='none', mew='2', mec='blue')
'''
############################################################################################################
# Plotting
plt.semilogy(rabiA_final, T1_final, 's', mfc='none', mew='2', mec='blue')
plt.xlim([0, 10])
plt.ylim([0, 3000])
plt.tick_params(labelsize=18)
plt.show()
