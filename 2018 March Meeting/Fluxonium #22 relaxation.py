import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap

# plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.figure(figsize=[5,5])
#Define file directory
directory = "G:\Projects\Fluxonium\Data\Simulation"
simulation = "#22 Relaxation"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[20,10])
#######################################################################################
N = 50
E_l = 0.4
E_c = 0.79
E_j = 3.4
level_num = 15
kB=1.38e-23

iState = 0
fState = 1
phi_ext = np.linspace(0.0,0.5,501)
p_element = np.zeros(len(phi_ext))
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
gamma_cap = np.zeros(len(phi_ext))
gamma_ind = np.zeros(len(phi_ext))
gamma_qp = np.zeros(len(phi_ext))
gamma_qp_array = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))
#'''
#######################################################################################
for idx, phi in enumerate(phi_ext):
    p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
    n_element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
    qp_element[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2.0 * np.pi, iState, fState))
    for idy in range(level_num):
        energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2.0*np.pi).eigenenergies()[idy]

np.savetxt(path + '_energies.txt', energies)
np.savetxt(path + '_chargeElement.txt', n_element)
np.savetxt(path + '_fluxElement.txt', p_element)
np.savetxt(path + '_qpElement.txt', qp_element)
#'''
#######################################################################################
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
w = energies[:,fState]-energies[:,iState]
T_diel=1.0e-3
thermal_factor_diel = (1+np.exp(-h*w*1e9/(kB*T_diel)))
# T_qp=100.0e-3
# thermal_factor_qp = (1+np.exp(-h*w*1e9/(kB*T_qp)))


for Q_cap in [0.5e6]:
    for idx in range(len(phi_ext)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap*5.0/w[idx]**0.7, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
    plt.semilogy(w, 1.0/gamma_cap *1e6, linewidth= 2.0, linestyle ='-')

#for x_qp in [20e-7,100e-7]:
#    Q_qp = 1.0/x_qp
#    for idx in range(len(phi_ext)):
#        gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp, w[idx], qp_element[idx])
#    # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
#    plt.semilogy(w, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='k')

# for T_qp in [0.25, 0.28]:
#     thermal_factor_qp = (1 + np.exp(-h * w * 1e9 / (kB * T_qp)))
#     for idx in range(len(phi_ext)):
#         gamma_qp[idx] = r_qp_finiteT(E_l, E_c, E_j, w[idx], qp_element[idx], T_qp)*thermal_factor_qp[idx]
#     # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='r')

# for x_qp in [10e-9]:
#     Q_qp = 1.0 / x_qp
#     for idx in range(len(phi_ext)):
#         gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx])
#     plt.semilogy(w, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--', color = 'orange')
    # plt.semilogy(w, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')

# for T_qp in [0.25]:
#     thermal_factor_qp = (1 + np.exp(-h * w * 1e9 / (kB * T_qp)))
#     for idx in range(len(phi_ext)):
#         gamma_qp_array[idx] = r_qp_array_finiteT(E_l, E_c, E_j, w[idx], p_element[idx], T_qp)*thermal_factor_qp[idx]
#     plt.semilogy(w, 1.0/(gamma_qp_array)*1e6, linewidth = 2.0, linestyle='--', color = 'orange')
    # plt.semilogy(w, 1.0/(gamma_qp_array+gamma_cap+gamma_qp)*1e6, linewidth = 2.0, linestyle='-.', color='k')

############################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #22\Summary'
# fname='T1T2 0mA.txt'
fname = 'T1T2 96mA.txt'
path = directory + '\\'+ fname
data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
current = data[:,0]
freq = data[:,1]
T1 = data[:,2]
matrix_element_interp = np.interp(freq, w, p_element, period = 3.8)
# plt.errorbar((current-0.292)*0.5/0.511, T1, fmt='s', mfc='none', mew=2.0, mec='blue')
# plt.errorbar((current-96.157)*0.5/0.511, T1, fmt='s', mfc='none', mew=2.0, mec='blue')
plt.errorbar(freq, T1, fmt='s', mfc='none', mew=2.0, mec='blue')
############################################################################
# plt.grid()
# plt.xticks([0.0,0.5])
# plt.yticks([10,100])
# plt.yticks([10,100,1000])
plt.yticks([])
# plt.ylim([2,1000])
# plt.xlim(0.45,0.55)
plt.tick_params(labelsize=18)
plt.show()