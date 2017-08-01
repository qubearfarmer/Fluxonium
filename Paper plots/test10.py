from matplotlib import pyplot as plt
import numpy as np
from qutip import*

plt.rc('font', family='serif')
plt.rc('text', usetex=False)
from Fluxonium_hamiltonians.Squid_small_junctions import coupled_hamiltonian

plt.figure(figsize=[7,4])
#############################################################################################################################
directory = "G:\Projects\Fluxonium\Data\Fluxonium #10"
fname = "One tune spectroscopy_YOKO 0mAto50mA_ qubit tone off_Cav_10p30GHz to 10p32GHz_1dBm_pulse_4000_2800_after 2nd thermal cycle"
path = directory + "\\" + fname

freq = np.genfromtxt(path+'_Freq.csv', delimiter = ',')/1e9
current = np.genfromtxt(path + '_Current.csv', delimiter = ',')*1e3
mag = np.genfromtxt(path + '_Mag.csv', delimiter = ',')
wr = 10.304
freq = freq - wr
freq = freq*1e3
X,Y = np.meshgrid(current, freq)
Z = mag.transpose()

plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r')
plt.xlim([current[0], 46])
plt.ylim([freq[0], freq[-1]])

directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Dispersive_shifts_wSquid"
path = directory + "\\" + simulation

#############################################################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[20,5])
#Qubit and computation parameters
N = 40
Nr = 3
E_l=0.735773762652
E_c=0.537375025825
E_j_sum=22.3
A_j=3.83424869313e-12
A_c=1.46689233147e-10
d=0.185865262485
beta_squid=-2.58488114861e-05
beta_ext=-0.0251115059548
B_coeff = 60
g=0.092


level_num = 10

current = np.linspace(0, 0.05, 501)
energies = np.zeros((len(current),level_num))
sim_dat = np.zeros((len(current),3))
'''
#Compute eigenenergies
for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = coupled_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                            2 * np.pi * (flux_ext / phi_o - beta_ext), Nr, wr, g)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

np.savetxt(path + "_energies.txt", energies)
'''
energies = np.genfromtxt(path + "_energies.txt")

#Plot transition energies
for idx in range(1, level_num):
    plt.plot(current*1e3, (energies[:,idx]-energies[:,0] - wr)*1e3, 'k-', linewidth = 1.0)
# for idx in range(1, level_num):
    # plt.plot(current*1e3, energies[:,idx], 'k-')
plt.tick_params(labelsize=18.0)
plt.show()