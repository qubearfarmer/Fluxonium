# Analyzing T1 data around 38p6mA for 0-2 transition
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem

plt.rc('font', family='serif')

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

fig, ax1 = plt.subplots(figsize=(10, 4.5))
# ax1.set_xticks([38.55, 38.6, 38.65])
ax1.tick_params(labelsize=18)
##################################################################################################################
############################################# Spectrum ###########################################################
##################################################################################################################

import matplotlib.cm as cm
from qutip import *

#####################################################################################################################################################################################
#####################################################################################################################################################################################
# Plasmon line scan
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_0to20mA_currentMode_qubit_n30dBm_cav_1dBm_avg50K_pulse25'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Current = np.genfromtxt(path_cur, delimiter=',') * 1e3
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Current) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        I = Current[idx:idx + 11]
        X, Y = np.meshgrid(I, f)
        ax1.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin=-5, vmax=0)
########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_20to30mA_currentMode_qubit_n5dBm_cav_5dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Current = np.genfromtxt(path_cur, delimiter=',') * 1e3
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Current) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        I = Current[idx:idx + 11]
        X, Y = np.meshgrid(I, f)
        ax1.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin=-5, vmax=0)
########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_30to32mA_currentMode_qubit_n5dBm_cav_5dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Current = np.genfromtxt(path_cur, delimiter=',') * 1e3
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Current) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        I = Current[idx:idx + 11]
        X, Y = np.meshgrid(I, f)
        ax1.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin=-5, vmax=0)
########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_32to39mA_currentMode_qubit_n5dBm_cav_5dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Current = np.genfromtxt(path_cur, delimiter=',') * 1e3
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Current) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        I = Current[idx:idx + 11]
        X, Y = np.meshgrid(I, f)
        ax1.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin=-5, vmax=0)
########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_39to50mA_currentMode_qubit_0dBm_cav_1dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Current = np.genfromtxt(path_cur, delimiter=',') * 1e3
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Current) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        I = Current[idx:idx + 11]
        X, Y = np.meshgrid(I, f)
        ax1.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin=-6, vmax=-2)

########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_46to48mA_currentMode_qubit_0dBm_cav_5dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Current = np.genfromtxt(path_cur, delimiter=',') * 1e3
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Current) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        I = Current[idx:idx + 11]
        X, Y = np.meshgrid(I, f)
        ax1.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin=-4, vmax=-1)
# Small scan
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_43to44mA_currentMode_qubit_2p5to3p2GHz_0dBm_cav5dBm_avg50K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_current = directory + '\\' + measurement + '_I.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
I = np.genfromtxt(path_current, delimiter=',') * 1e3
Z = RawData.transpose()
X, Y = np.meshgrid(I, Freq)
ax1.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin=-4, vmax=-1)
########################################################################
# Small scan
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_43p15to43p85mA_currentMode_qubit_1p5to2p5GHz_0dBm_cav5dBm_avg50K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_current = directory + '\\' + measurement + '_I.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
I = np.genfromtxt(path_current, delimiter=',') * 1e3
Z = RawData.transpose()
X, Y = np.meshgrid(I, Freq)
ax1.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin=-4, vmax=-1)
########################################################################
# Blue side band
directory = 'C:\Data\Fluxonium #10'
measurement = 'Two tune spectroscopy_YOKO 43p4to43p6mA_ qubit tone 10p5to11p2GHz_5dBm_Cav_10p304GHz_8dBm_pulse 34us duty2_avg5K'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_current = directory + '\\' + measurement + '_I.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
I = (np.genfromtxt(path_current, delimiter=',') - 0.00003) * 1e3
Z = RawData.transpose()
X, Y = np.meshgrid(I, Freq)
ax1.pcolormesh(X, Y, Z, cmap=cm.RdBu_r, vmin=-5, vmax=5)
########################################################################
# Red side band
directory = 'C:\Data\Fluxonium #10'
measurement = 'Two tune spectroscopy_YOKO 43p4to43p6mA_ qubit tone 8p5to10p2GHz_5dBm_Cav_10p304GHz_8dBm_pulse 34us duty2_avg5K'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_current = directory + '\\' + measurement + '_I.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
I = (np.genfromtxt(path_current, delimiter=',') - 0.00003) * 1e3
Z = RawData.transpose()
X, Y = np.meshgrid(I, Freq)
ax1.pcolormesh(X, Y, Z, cmap=cm.RdBu_r, vmin=-5, vmax=5)

# Plotting data taken with new software
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_41to43mA_Qubit_3to4GHz_5dBm_Cav_10.3039GHz_8dBm_IF_0.05GHz_measTime_500ns_avg_50000'
path = directory + '\\' + measurement

# Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1::] - 0.04
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::, 0]  # phase is recorded in rad
phase = phase  #
mag = data[1::, 0]

# plt.figure(1)
Z = np.zeros((len(current), len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx * len(freq):(idx + 1) * len(freq)])
    Z[idx, :] = temp - np.average(temp)
Z = Z * 180 / (np.pi)
X, Y = np.meshgrid(current, freq[0:len(freq) / 2 + 2])
Z1 = Z.transpose()[0:len(freq) / 2 + 2]
ax1.pcolormesh(X, Y, Z1, cmap='GnBu_r', vmin=-4, vmax=-1, alpha=1)

X, Y = np.meshgrid(current, freq[len(freq) / 2 + 2:len(freq) - 1])
Z2 = Z.transpose()[len(freq) / 2 + 2:len(freq) - 1]
ax1.pcolormesh(X, Y, Z2, cmap='GnBu_r', vmin=-4, vmax=-1, alpha=1)

########################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_38.1to40mA_Qubit_3.5to5GHz_5dBm_Cav_10.3039GHz_8dBm_IF_0.05GHz_measTime_500ns_avg_25000'
path = directory + '\\' + measurement

# Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1::] - 0.04
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::, 0]  # phase is recorded in rad
phase = phase  #
mag = data[1::, 0]

Z = np.zeros((len(current), len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx * len(freq):(idx + 1) * len(freq)])
    Z[idx, :] = temp - np.average(temp)
Z = Z * 180 / (np.pi)
Z = Z.transpose()[1:len(freq) - 1]
X, Y = np.meshgrid(current, freq[1:len(freq) - 1])
# plt.figure(1)
ax1.pcolormesh(X, Y, Z, cmap='GnBu_r', vmin=-4, vmax=-1, alpha=1)

#####################################################################
# high power scan
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_38to40mA_Qubit_3.5to5GHz_10dBm_Cav_10.3045GHz_5dBm_IF_0.05GHz_measTime_500ns_avg_50000'
path = directory + '\\' + measurement

# Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1::] - 0.04
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::, 0]  # phase is recorded in rad
phase = phase  #
mag = data[1::, 0]

Z = np.zeros((len(current), len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx * len(freq):(idx + 1) * len(freq)])
    Z[idx, :] = temp - np.average(temp)
Z = Z * 180 / (np.pi)
Z = Z.transpose()[1:len(freq) - 1]
X, Y = np.meshgrid(current, freq[1:len(freq) - 1])
# plt.figure(1)
# plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = -4, vmax=-1, alpha = 1)

#####################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_38.56to38.66mA_Qubit_4.2to5.1GHz_-6dBm_Cav_10.3045GHz_5dBm_IF_0.05GHz_measTime_500ns_avg_20000'
path = directory + '\\' + measurement

# Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1:-1] - 0.04
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::, 0]  # phase is recorded in rad
phase = phase  #
mag = data[1::, 0]

Z = np.zeros((len(current), len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx * len(freq):(idx + 1) * len(freq)])
    Z[idx, :] = temp - np.average(temp)
Z = Z * 180 / (np.pi)
Z = Z.transpose()[1:len(freq) - 1]
X, Y = np.meshgrid(current, freq[1:len(freq) - 1])

# plt.pcolormesh(X,Y,Z, cmap= 'Reds_r', vmin = -4, vmax=-0.5, alpha = 0.2)
#################################################################################################
# Fine scan
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_38.56to38.66mA_Qubit_4.2to5.1GHz_-6dBm_Cav_10.3045GHz_5dBm_IF_0.05GHz_measTime_500ns_avg_20000'
path = directory + '\\' + measurement

# Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1:-1] - 0.037
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::, 0]  # phase is recorded in rad
phase = phase  #
mag = data[1::, 0]

Z = np.zeros((len(current), len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx * len(freq):(idx + 1) * len(freq)])
    Z[idx, :] = temp - np.average(temp)
Z = Z * 180 / (np.pi)
Z = Z.transpose()[1:len(freq) - 1]
X, Y = np.meshgrid(current, freq[1:len(freq) - 1])
ax1.pcolormesh(X, Y, Z, cmap='GnBu_r', vmin=-4, vmax=-1)
#####################################################################################################################################################################################
#####################################################################################################################################################################################

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

"""
First section of the script attempts to plot the energies vs external flux
"""


# Hamiltonian definition
def Ho(N, E_l, E_c, E_j_sum, d, phi_squid, phi_ext):
    E_j1 = 0.5 * E_j_sum * (1 + d)
    E_j2 = 0.5 * E_j_sum * (1 - d)
    a = tensor(destroy(N))
    mass = 1.0 / (8.0 * E_c)
    w = sqrt(8.0 * E_c * E_l)
    phi = (a + a.dag()) * (8 * E_c / E_l) ** (0.25) / np.sqrt(2)
    na = 1j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2)
    ope1 = 1j * (-phi + phi_ext)
    ope2 = 1j * (
        phi - phi_ext + phi_squid)  # phi_squid and phi_ext here are the external phases, or normalized flux, = flux*2pi/phi_o
    H = 4.0 * E_c * na ** 2 + 0.5 * E_l * (phi) ** 2 - 0.5 * E_j1 * (ope1.expm() + (-ope1).expm()) - 0.5 * E_j2 * (
        ope2.expm() + (-ope2).expm())
    return H.eigenenergies()


def coupled_H(Na, E_l, E_c, E_j_sum, d, phi_squid, phi_ext, Nr, wr, g):
    E_j1 = 0.5 * E_j_sum * (1 + d)
    E_j2 = 0.5 * E_j_sum * (1 - d)
    a = tensor(destroy(Na), qeye(Nr))
    b = tensor(qeye(Na), destroy(Nr))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    ope1 = 1.0j * (phi_ext - phi)
    ope2 = 1.0j * (phi + phi_squid - phi_ext)
    H_f = 4.0 * E_c * na ** 2 + 0.5 * E_l * (phi) ** 2 - 0.5 * E_j1 * (ope1.expm() + (-ope1).expm()) - 0.5 * E_j2 * (
        ope2.expm() + (-ope2).expm())
    H_r = wr * (b.dag() * b + 1.0 / 2)
    H_c = -2 * g * na * (b.dag + b)
    H = H_f + H_r + H_c
    return H.eigenenergies()


def trans_energies(N, E_l, E_c, E_j_sum, d, A_j, A_c, B_coeff, beta_squid, beta_ext, level_num, current, iState):
    B_field = current * B_coeff * 1e-4  # in T, this depends on a seperate measurement
    phi_squid = B_field * A_j  # these are flux, not normalized
    phi_ext = B_field * A_c
    trans_energy = np.zeros((level_num - iState, len(phi_ext)))
    for idx in range(len(phi_ext)):
        energies = Ho(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (phi_squid[idx] / phi_o - beta_squid),
                      2 * np.pi * (phi_ext[idx] / phi_o - beta_ext))  # normalize the flux -> phase here
        for level in range(iState + 1, level_num):
            trans_energy[level - iState, idx] = energies[level] - energies[iState]
    return trans_energy


def coupled_trans_energies(N, E_l, E_c, E_j_sum, d, A_j, A_c, B_coeff, beta_squid, beta_ext, level_num, current, iState,
                           Nr, wr, g):
    B_field = current * B_coeff * 1e-4  # in T, this depends on a seperate measurement
    phi_squid = B_field * A_j  # these are flux, not normalized
    phi_ext = B_field * A_c
    trans_energy = np.zeros((level_num - iState, len(phi_ext)))
    for idx in range(len(phi_ext)):
        energies = coupled_H(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (phi_squid[idx] / phi_o - beta_squid),
                             2 * np.pi * (phi_ext[idx] / phi_o - beta_ext), Nr, wr,
                             g)  # normalize the flux -> phase here
        for level in range(iState + 1, level_num):
            trans_energy[level - iState, idx] = energies[level] - energies[iState]
    return trans_energy


########################################################################
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

current = np.linspace(0.037, 0.039, 201)
B_coeff = 60
level_num = 5

iState = 0
spectrum = trans_energies(N, E_l, E_c, E_j_sum, d, A_j, A_c, B_coeff, beta_squid, beta_ext, level_num, current, iState)
ax1.plot(current * 1e3, spectrum[2, :], linewidth=1, color='black', linestyle='-', alpha=0.2)

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
# '''
# current = flux * 1e-3
# # current = np.linspace(0.038,0.046, 801)
# N = 50
# E_l = 0.722729827116
# E_c = 0.552669197076
# E_j_sum = 17.61374383
# A_j = 4.76321410213e-12
# A_c = 1.50075181762e-10
# d = 0.125005274368
# beta_squid = 0.129912406349
# beta_ext = 0.356925557542
#
# B_coeff = 60
# level_num = 5
# energies = np.zeros((len(current), level_num))
# qp_element = np.zeros((len(current), 2))
# n_element = np.zeros(len(current))
# p_element = np.zeros(len(current))
#
# iState = 1
# fState = 2
# for idx, curr in enumerate(current):
#     flux_squid = curr * B_coeff * A_j * 1e-4
#     flux_ext = curr * B_coeff * A_c * 1e-4
#     H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
#                          2 * np.pi * (flux_ext / phi_o - beta_ext))
#     for idy in range(level_num):
#         energies[idx, idy] = H.eigenenergies()[idy]

T1_final = []
T1_err_final = []
flux_final = []
freq_final = []
# trans_energy = energies[:, fState] - energies[:, iState]
# plt.plot(current*1e3, energies[:,2]-energies[:,0])
for idx in range(len(T1)):
    if Rabi_A[idx] > 0 and flux[idx] > 38.2 and flux[idx] < 38.8:
        T1_final = np.append(T1_final, T1[idx])
        T1_err_final = np.append(T1_err_final, T1_err[idx])
        flux_final = np.append(flux_final, flux[idx])
        freq_final = np.append(freq_final, freq[idx])
# '''
print len(T1_final)
ax1.set_ylim([4.4, 8.6])
ax1.set_xlim([38.2, 38.75])
ax1.set_xticks([38.3, 38.4, 38.5, 38.6, 38.7])
ax2 = ax1.twinx()
ax = plt.gca()
ax.set_yscale('log')
ax2.errorbar(flux_final, T1_final, yerr=T1_err_final, fmt='d', mfc='none', mew='2', mec='red', ecolor='red')
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax2.set_xlim([38.2, 38.8])
ax2.set_ylim([5, 1000])
ax2.tick_params(labelsize=18)
#########################################################################################
################################### T1 simulation ######################################
#########################################################################################
# '''
current = np.linspace(38.2, 38.8, 61) * 1e-3
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
########################################Upper limit########################################
iState = 0
fState = 2
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
#############################################################
hbar = h / (2 * np.pi)
kB = 1.38064852e-23
T = 1e-2
E_c = E_c / 1.509190311677e+24  # convert GHz to J
E_l = E_l / 1.509190311677e+24  # convert to J
E_j_sum = E_j_sum / 1.509190311677e+24  # convert to J
E_j1 = 0.5 * E_j_sum * (1 + d)
E_j2 = 0.5 * E_j_sum * (1 - d)
delta_alum = 5.447400321e-23  # J

Q_cap = 0.7e6
Q_ind = 0.8e6
Q_qp = 17e6
# Q_cap = 0.6e5
# Q_ind = 0.8e6
# Q_qp = 0.8e6

cap = e ** 2 / (2.0 * E_c)
ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
gk = e ** 2.0 / h
g1 = 8.0 * E_j1 * gk / delta_alum
g2 = 8.0 * E_j2 * gk / delta_alum
#######################################
trans_energy = energies[:, fState] - energies[:, iState]
w = trans_energy * 1e9 * 2 * np.pi
# w = 3e9 * 2 * np.pi
Y_cap = w * cap / Q_cap
Y_ind = 1.0 / (w * ind * Q_ind)
Y_qp1 = (g1 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)
Y_qp2 = (g2 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)

gamma_cap = np.zeros(len(current))
gamma_qp = np.zeros((len(current), 2))

for idx in range(len(current)):
    gamma_cap[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_cap[idx] * (
        1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    # gamma_ind[idx] = (phi_o * pem_sim[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_ind * (1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    gamma_qp[idx, 0] = (qp_element[idx, 0]) ** 2 * (w[idx] / np.pi / gk) * Y_qp1[idx]
    gamma_qp[idx, 1] = (qp_element[idx, 1]) ** 2 * (w[idx] / np.pi / gk) * Y_qp2[idx]

##################################################1-2#############################################################
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
    # n_element[idx] = nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
    #                      2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    p_element[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element[idx, :] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
#############################################################
hbar = h / (2 * np.pi)
kB = 1.38064852e-23
T = 1e-2
E_c = E_c / 1.509190311677e+24  # convert GHz to J
E_l = E_l / 1.509190311677e+24  # convert to J
E_j_sum = E_j_sum / 1.509190311677e+24  # convert to J
E_j1 = 0.5 * E_j_sum * (1 + d)
E_j2 = 0.5 * E_j_sum * (1 - d)
delta_alum = 5.447400321e-23  # J

cap = e ** 2 / (2.0 * E_c)
ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
gk = e ** 2.0 / h
g1 = 8.0 * E_j1 * gk / delta_alum
g2 = 8.0 * E_j2 * gk / delta_alum
#######################################
trans_energy = energies[:, fState] - energies[:, iState]
w = trans_energy * 1e9 * 2 * np.pi
# w = 3e9 * 2 * np.pi
Y_cap = w * cap / Q_cap
Y_ind = 1.0 / (w * ind * Q_ind)
Y_qp1 = (g1 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)
Y_qp2 = (g2 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)

gamma_cap12 = np.zeros(len(current))
gamma_qp12 = np.zeros((len(current), 2))

for idx in range(len(current)):
    gamma_cap12[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_cap[idx] * (
        1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    # gamma_ind[idx] = (phi_o * pem_sim[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_ind * (1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    gamma_qp12[idx, 0] = (qp_element[idx, 0]) ** 2 * (w[idx] / np.pi / gk) * Y_qp1[idx]
    gamma_qp12[idx, 1] = (qp_element[idx, 1]) ** 2 * (w[idx] / np.pi / gk) * Y_qp2[idx]

# T1_sim = 1 / (gamma_cap + gamma_cap12)
# ax2.semilogy(current * 1e3, T1_sim * 1e6, linewidth='2', color='black', linestyle='--')
T1_sim = 1 / (gamma_qp[:, 0] + gamma_qp[:, 1] + gamma_qp12[:, 0] + gamma_qp12[:, 1])
ax2.semilogy(current * 1e3, T1_sim * 1e6, linewidth='2', color='black', linestyle='--')

# T1_sim = 1 / (gamma_qp[:, 0] + gamma_qp[:, 1])
# ax2.semilogy(current * 1e3, T1_sim * 1e6, linewidth='2', color='black', linestyle='-.')
########################################Lower limit########################################
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
fState = 2
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
#############################################################
hbar = h / (2 * np.pi)
kB = 1.38064852e-23
T = 1e-2
E_c = E_c / 1.509190311677e+24  # convert GHz to J
E_l = E_l / 1.509190311677e+24  # convert to J
E_j_sum = E_j_sum / 1.509190311677e+24  # convert to J
E_j1 = 0.5 * E_j_sum * (1 + d)
E_j2 = 0.5 * E_j_sum * (1 - d)
delta_alum = 5.447400321e-23  # J

# Q_cap = 0.7e6
# Q_ind = 0.8e6
# Q_qp = 17e6
Q_cap = 0.6e5
Q_ind = 0.8e6
Q_qp = 0.8e6

cap = e ** 2 / (2.0 * E_c)
ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
gk = e ** 2.0 / h
g1 = 8.0 * E_j1 * gk / delta_alum
g2 = 8.0 * E_j2 * gk / delta_alum
#######################################
trans_energy = energies[:, fState] - energies[:, iState]
w = trans_energy * 1e9 * 2 * np.pi
# w = 3e9 * 2 * np.pi
Y_cap = w * cap / Q_cap
Y_ind = 1.0 / (w * ind * Q_ind)
Y_qp1 = (g1 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)
Y_qp2 = (g2 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)

gamma_cap = np.zeros(len(current))
gamma_qp = np.zeros((len(current), 2))

for idx in range(len(current)):
    gamma_cap[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_cap[idx] * (
        1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    # gamma_ind[idx] = (phi_o * pem_sim[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_ind * (1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    gamma_qp[idx, 0] = (qp_element[idx, 0]) ** 2 * (w[idx] / np.pi / gk) * Y_qp1[idx]
    gamma_qp[idx, 1] = (qp_element[idx, 1]) ** 2 * (w[idx] / np.pi / gk) * Y_qp2[idx]

##################################################1-2#############################################################
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
    # n_element[idx] = nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
    #                      2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    p_element[idx] = pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    qp_element[idx, :] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                              2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
#############################################################
hbar = h / (2 * np.pi)
kB = 1.38064852e-23
T = 1e-2
E_c = E_c / 1.509190311677e+24  # convert GHz to J
E_l = E_l / 1.509190311677e+24  # convert to J
E_j_sum = E_j_sum / 1.509190311677e+24  # convert to J
E_j1 = 0.5 * E_j_sum * (1 + d)
E_j2 = 0.5 * E_j_sum * (1 - d)
delta_alum = 5.447400321e-23  # J

cap = e ** 2 / (2.0 * E_c)
ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
gk = e ** 2.0 / h
g1 = 8.0 * E_j1 * gk / delta_alum
g2 = 8.0 * E_j2 * gk / delta_alum
#######################################
trans_energy = energies[:, fState] - energies[:, iState]
w = trans_energy * 1e9 * 2 * np.pi
# w = 3e9 * 2 * np.pi
Y_cap = w * cap / Q_cap
Y_ind = 1.0 / (w * ind * Q_ind)
Y_qp1 = (g1 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)
Y_qp2 = (g2 / (2 * Q_qp)) * (2 * delta_alum / (hbar * w)) ** (1.5)

gamma_cap12 = np.zeros(len(current))
gamma_qp12 = np.zeros((len(current), 2))

for idx in range(len(current)):
    gamma_cap12[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_cap[idx] * (
        1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    # gamma_ind[idx] = (phi_o * pem_sim[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_ind * (1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    gamma_qp12[idx, 0] = (qp_element[idx, 0]) ** 2 * (w[idx] / np.pi / gk) * Y_qp1[idx]
    gamma_qp12[idx, 1] = (qp_element[idx, 1]) ** 2 * (w[idx] / np.pi / gk) * Y_qp2[idx]

# T1_sim = 1 / (gamma_cap + gamma_cap12)
# ax2.semilogy(current * 1e3, T1_sim * 1e6, linewidth='2', color='black', linestyle='--')
T1_sim = 1 / (gamma_qp[:, 0] + gamma_qp[:, 1] + gamma_qp12[:, 0] + gamma_qp12[:, 1])
ax2.semilogy(current * 1e3, T1_sim * 1e6, linewidth='2', color='black', linestyle='--')

# T1_sim = 1 / (gamma_qp[:, 0] + gamma_qp[:, 1])
# ax2.semilogy(current * 1e3, T1_sim * 1e6, linewidth='2', color='black', linestyle='-.')

#########################################################################################
plt.show()
