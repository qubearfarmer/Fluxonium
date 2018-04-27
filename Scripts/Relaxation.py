from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp_array as r_qp_array
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap_Z as r_cap_Z
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_fluxNoise as r_flux

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.figure(figsize=[10,7])
#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Relaxation_28"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.626e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[20,10])
#######################################################################################
N = 50
E_l = 1.128
E_c = 0.847
E_j = 4.79
level_num = 15

kB=1.38e-23
T=60.0e-3

iState = 0
fState = 1
phi_ext = np.linspace(-0.05,0.55,601)
p_element = np.zeros(len(phi_ext))
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
gamma_cap = np.zeros(len(phi_ext))
gamma_flux = np.zeros(len(phi_ext))
gamma_qp = np.zeros(len(phi_ext))
gamma_qp_array = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))

#######################################################################################
# for idx, phi in enumerate(phi_ext):
#     p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
#     n_element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
#     qp_element[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2.0 * np.pi, iState, fState))
#     for idy in range(level_num):
#         energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2.0*np.pi).eigenenergies()[idy]
#
# np.savetxt(path + '_energies.txt', energies)
# np.savetxt(path + '_chargeElement.txt', n_element)
# np.savetxt(path + '_fluxElement.txt', p_element)
# np.savetxt(path + '_qpElement.txt', qp_element)
#######################################################################################

energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
w = energies[:,fState]-energies[:,iState]
thermal_factor = (1+np.exp(-h*w*1e9/(kB*T)))
# plt.plot(w,n_element)

for Q_cap in [0.7e6]:
    for idx in range(len(phi_ext)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap, w[idx], p_element[idx])
    # plt.semilogy(phi_ext, 1.0 / gamma_cap * 1e6, linewidth=2.0, linestyle='-')
    # plt.semilogy(w, 1.0/gamma_cap *1e6, linewidth= 1.0, linestyle ='-',alpha = 0.5)
    plt.semilogy(w, 1.0/gamma_cap *1e6/thermal_factor, linewidth= 2.0, linestyle ='-')
# for A in [2e-6*phi_o]:
#     for idx in range(len(phi_ext)):
#         gamma_flux[idx] = r_flux(E_l, E_c, E_j, A, w[idx], p_element[idx])
#     plt.semilogy(phi_ext, 1.0/gamma_flux *1e6, linewidth= 2.0, linestyle ='-')

for x_qp in [20e-7]:
    Q_qp = 1.0/x_qp
    for idx in range(len(phi_ext)):
        gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp, w[idx], qp_element[idx])
    plt.semilogy(w, 1.0/(gamma_qp)*1e6/thermal_factor, linewidth = 2.5, linestyle='--')
#
for x_qp in [10e-9]:
    Q_qp = 1.0 / x_qp
    for idx in range(len(phi_ext)):
        gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx])
    # plt.semilogy(w, 1.0 / (gamma_qp_array) * 1e6 , linewidth=1.0, linestyle='-.',alpha = 0.5)
    plt.semilogy(w, 1.0 / (gamma_qp_array) * 1e6/thermal_factor, linewidth=2.0, linestyle='-.')
    # plt.semilogy(phi_ext, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')
#
#################################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #28\Summary'

# measurement = 'T1_summary_2018_04_13.txt'
measurement = 'T1summary0mA 4018_25_04.txt'
path = directory + '\\' + measurement
data = np.genfromtxt(path)
current = data[1:,0]
freq = data[1:,1]
T1 = data[1:,2]
T1_err = data[1:,3]
T1_sp = data[1:,4]
T1_sp_err = data[1:,5]
plt.errorbar(freq, T1, yerr = T1_err, fmt = 's', mfc = 'none', mew = 2.0, mec = 'b', ecolor = 'b')
plt.errorbar(freq, T1_sp, yerr = T1_sp_err, fmt = 'h', mfc = 'none', mew = 2.0, mec = 'm', ecolor = 'm')
plt.grid()

##################################################################################
plt.tick_params(labelsize=18)
# plt.ylim([5e1,1e3])
# plt.xlim([0.2,1.2])
plt.ylim([5e1,1e3])
plt.xlim([0.2,1.2])
plt.show()
