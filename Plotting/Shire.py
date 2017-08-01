"""
This script describes bare fluxonium spectrum
"""
from scipy import *
import numpy as np
from qutip import *
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Palatino']})
rc('text', usetex=True)
plt.close("all")
directory = 'G:\Old Server Data\GROUP\Shared\Projects\Fluxonium & qubits\Data\Fluxonium #10\Pulse'
measurement = 'S21_1tone_cav_10dBm_1p3to10p32GHz_YOKO_0to3V'
path_data = directory + '\\' + measurement + '_Mag.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_vol = directory + '\\' + measurement + '_Voltage.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
V = np.genfromtxt(path_vol, delimiter=',')
# V = np.linspace(-0.11,-0.09,201)
Z = RawData.transpose()

##Optional: calculate differential
Z_diff = np.diff(Z.transpose())
Z_diff = Z_diff.transpose()
###Plot the intensity map
X, Y = np.meshgrid(V, Freq)
plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r)  # , vmin = -1 , vmax = 1)
##Plot the intensity map
directory = 'G:\Old Server Data\GROUP\Shared\Projects\Fluxonium & qubits\Data\Fluxonium #10\Pulse'
measurement = 'S21_2tone_plasmonline_qubit_n30dBm_cav_10dBm&n30dB_YOKO_0to1p5V'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_vol = directory + '\\' + measurement + '_Voltage.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Voltage = np.genfromtxt(path_vol, delimiter=',')
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Voltage) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        V = Voltage[idx:idx + 11]
        X, Y = np.meshgrid(V, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r)

directory = 'G:\Old Server Data\GROUP\Shared\Projects\Fluxonium & qubits\Data\Fluxonium #10\Pulse'
measurement = 'S21_2tone_plasmonline_qubit_n27dBm_cav_10dBm&n30dB_YOKO_1p5to3V'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_vol = directory + '\\' + measurement + '_Voltage.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Voltage = np.genfromtxt(path_vol, delimiter=',')
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Voltage) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        V = Voltage[idx:idx + 11]
        X, Y = np.meshgrid(V, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r)

directory = 'G:\Old Server Data\GROUP\Shared\Projects\Fluxonium & qubits\Data\Fluxonium #10\Pulse'
measurement = 'S21_2tone_plasmonline_qubit_n20dBm_cav_10dBm&n30dB_YOKO_3to4V'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_vol = directory + '\\' + measurement + '_Voltage.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Voltage = np.genfromtxt(path_vol, delimiter=',')
# Voltage = np.linspace(3,4,1000)
for idx in range(len(Voltage) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        V = Voltage[idx:idx + 11]
        X, Y = np.meshgrid(V, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r)

directory = 'G:\Old Server Data\GROUP\Shared\Projects\Fluxonium & qubits\Data\Fluxonium #10\Pulse'
measurement = 'S21_2tone_plasmonline_qubit_n10dBm_cav_10dBm&n30dB_YOKO_4to5'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_vol = directory + '\\' + measurement + '_Voltage.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Voltage = np.genfromtxt(path_vol, delimiter=',')
# Voltage = np.linspace(3,4,1000)
for idx in range(len(Voltage) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        V = Voltage[idx:idx + 11]
        X, Y = np.meshgrid(V, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r)
directory = 'G:\Old Server Data\GROUP\Shared\Projects\Fluxonium & qubits\Data\Fluxonium #10\Pulse'
measurement = 'S21_2tone_plasmonline_qubit_n10dBm_cav_10dBm&n30dB_YOKO_4to5 cont'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_vol = directory + '\\' + measurement + '_Voltage.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Voltage = np.genfromtxt(path_vol, delimiter=',')
# Voltage = np.linspace(3,4,1000)
for idx in range(len(Voltage) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        V = Voltage[idx:idx + 11]
        X, Y = np.meshgrid(V, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r)

directory = 'G:\Old Server Data\GROUP\Shared\Projects\Fluxonium & qubits\Data\Fluxonium #10\Pulse'
measurement = 'S21_2tone_plasmonline2_qubit_n5dBm_cav_1dBm&n30dB_YOKO_0to3V'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_vol = directory + '\\' + measurement + '_Voltage.csv'

RawData = np.genfromtxt(path_data, delimiter=',')
Freq = np.genfromtxt(path_freq, delimiter=',') / 1e9
Voltage = np.genfromtxt(path_vol, delimiter=',')
# Voltage = np.linspace(0,3,3000)
for idx in range(len(Voltage) - 1):
    if (idx % 10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx + 11].transpose()
        V = Voltage[idx:idx + 11]
        X, Y = np.meshgrid(V, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r)

# Define constants
e = 1.602176e-19  # Fundamental charge
h = 6.62607e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum
N = 60
M = 3
a = tensor(destroy(N), qeye(M))  # cavity
bb = tensor(qeye(N), destroy(M))  # atom
"""
First section of the script attempts to plot the energies vs external flux
"""


# Hamiltonian definition
def Ho(N, E_l, E_c, E_j, phi_ext, wcav, g):
    # a = tensor(destroy(N))
    # bb = tensor(destroy(M))
    mass = 1.0 / (8.0 * E_c)
    w = sqrt(8.0 * E_c * E_l)
    phi = (a + a.dag()) * (8 * E_c / E_l) ** (0.25) / np.sqrt(2)
    na = 1j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2)
    ope = 1j * (phi + phi_ext)
    H = 4.0 * E_c * na ** 2 + 0.5 * E_l * phi ** 2 - 0.5 * E_j * (ope.expm() + (-ope).expm()) - g * (
    bb + bb.dag()) * na + wcav * bb.dag() * bb
    # H =  - 0.5*E_j*(ope.expm() + (-ope).expm())+g*(bb+bb.dag())*(a + a.dag()) +wcav*bb.dag()*bb
    return H.eigenenergies()


# Define spectrum plotting wrt phi_ext
def plot_energies(N, E_l, E_c, E_j, phi_ext, level_num, voltage, wcav, g):
    energy = np.empty((level_num, len(phi_ext)), dtype=float)
    for idx in range(len(phi_ext)):
        energies = Ho(N, E_l, E_c, E_j[idx], phi_ext[idx] * 2 * np.pi / phi_o, M, wcav, g)
        offset = Ho(N, E_l, E_c, E_j[idx], 0, M, wcav, g)[0]
        for level in range(level_num):
            energy[level, idx] = energies[level] - offset
    for idx in range(level_num):
        line = plt.plot(voltage, energy[idx, :])
        plt.setp(line, linewidth=2.0)
    return


def plot_trans_energies(N, E_l, E_c, E_j, phi_ext, level_num, voltage, wcav, g):
    energy0 = np.empty((level_num, len(phi_ext)), dtype=float)
    energy1 = np.empty((level_num, len(phi_ext)), dtype=float)
    energy2 = np.empty((level_num, len(phi_ext)), dtype=float)
    ejs = np.empty((level_num, len(phi_ext)), dtype=float)
    for idx in range(len(phi_ext)):
        energies = Ho(N, E_l, E_c, E_j[idx], phi_ext[idx] * 2 * np.pi / phi_o, wcav, g)
        # 1 photon
        for level in range(0, level_num):
            energy0[level, idx] = energies[level] - energies[0]
            # energy0[level,idx]=energies[level]
            energy1[level, idx] = energies[level] - energies[1]
            energy2[level, idx] = energies[level] - energies[2]
            ejs[level, idx] = E_j[idx]
    for idx in range(0, level_num):
        # line = plt.plot(voltage, energy0[idx,:]-9.874)
        # plt.setp(line,linewidth=2.0, linestyle ='-', color = "green", alpha=0.75)
        # line = plt.plot(voltage, energy0[idx,:]/2)
        # plt.setp(line,linewidth=2.0, linestyle ='-', color = "blue", alpha=0.6)
        # line = plt.plot(voltage, energy0[idx,:]/3)
        # plt.setp(line,linewidth=2.0, linestyle ='-', color = "purple", alpha=0.75)
        line = plt.plot(voltage, energy0[idx, :])
        plt.setp(line, linewidth=2.0, linestyle='-', color="red", alpha=0.6)

    return


# Use the plotting function with parameters (N,E_l, E_c, E_j_max, level_num)
# N =100
# M=5
wcav = 10.3045
# g=.25
E_l = .65
# Ejoec=40
wp = 9.81
E_c = 0.534
# E_j_max=sqrt(Ejoec/8)*wp
E_j_max = wp * wp / 8.0 / E_c
g = .21 * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2)
# E_c=.8
level_num = 10
coil_resistance = 125

# Define external parameters
voltage = linspace(0, 5, 1000)  # in V
V_res = 0.007
current = (voltage + V_res) / coil_resistance
B_coeff = 50.65
B_field = current * B_coeff * 1e-4  # in T, this depends on a seperate measurement
area_ratio = 55.5
# A_j = 5.09e-12  #in m
# A_c = A_j*area_ratio
A_j = 4.53  # in um^2
A_j_p = A_j * 1e-12 * 50.65 * 1e-4
loopupdate = 1.565
A_c = 177.25  # in um^2
A_c_p = A_c * 1e-12 * 50.65 * 1e-4
# phi_ext_junc = B_field*A_j
phi_ext_junc = current * A_j_p
# phi_ext_cir = B_field*A_c
phi_ext_cir = current * A_c_p
d = 0.07
E_j = abs(E_j_max * np.cos(np.pi * (phi_ext_junc) / phi_o) * sqrt(1 + (d * np.tan(np.pi * phi_ext_junc / phi_o)) ** 2))
# Use plotting function here
fig = plt.figure(1)
line = plot_trans_energies(N, E_l, E_c, E_j, phi_ext_cir, level_num, voltage, wcav, g)
plt.title(r'E_L=' + str(E_l) + r', E_C=' + str(E_c) + r', E_{J_{max}}=' + str(E_j_max) + '\n'
          + r'A_{squid}=' + str(A_j) + r'\mu m^2, A_{circuit}=' + str(A_c) + r'\mu m^2, d=' + str(
    d) + '\n' + r' g=' + str(g) + r', wcav=' + str(wcav))
plt.ylabel(r'Transition Energy (GHz)')
plt.xlabel(r'Voltage(V)')
plt.ylim([10.3, 10.35])
# plt.ylim([0,20])
plt.grid()
plt.xlim([0, 3])

plt.show()