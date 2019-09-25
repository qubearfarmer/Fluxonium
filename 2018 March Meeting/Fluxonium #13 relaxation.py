import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap

# plt.rc('text', usetex=True)
# plt.rc('font', family='sans-serif')
plt.figure(figsize=[7,5])
#Define file directory
directory = "C:\\Users\\nguyen89\Documents\\Fluxonium simulation"
simulation = "Relaxation_13"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[20,10])
#######################################################################################
N = 50
N = 50
B_coeff = 25
E_l=1.01
E_c=0.835
E_j=3
level_num = 15

kB=1.38e-23


iState = 0
fState = 1
phi_ext = np.linspace(0.0,0.51,501)
p_element = np.zeros(len(phi_ext))
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
gamma_cap = np.zeros(len(phi_ext))
gamma_ind = np.zeros(len(phi_ext))
gamma_qp = np.zeros(len(phi_ext))
gamma_qp_array = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))

#######################################################################################
# '''
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
# '''
#######################################################################################
phi_ext = np.genfromtxt(path+'_flux.txt')
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
w = energies[:,fState]-energies[:,iState]
T_diel = 50.0e-3
thermal_factor_diel = (1+np.exp(-h*w*1e9/(kB*T_diel)))
T_qp=100.0e-3
thermal_factor_qp = (1+np.exp(-h*w*1e9/(kB*T_qp)))
alpha =1

for Q_cap in [0.2e6]:
    for idx in range(len(phi_ext)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap*5.0/w[idx]**0.7, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
    # plt.semilogy(w**alpha, 1.0 / gamma_cap * 1e6 * p_element**2, linewidth=2.0, linestyle ='--', color='orange')
    # plt.semilogy(w ** alpha, 1.0 / gamma_cap * 1e6 , linewidth=2.0, linestyle='--', color='orange')
    # plt.semilogy(phi_ext, 1.0/gamma_cap *1e6, linewidth= 2.0, linestyle ='--', color='orange')

for x_qp in [10e-7]:
    Q_qp = 1.0/x_qp
    for idx in range(len(phi_ext)):
        gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp*5.0/w[idx], w[idx], qp_element[idx])
    plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
    # plt.semilogy(phi_ext, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='k')
#
# for T_qp in [0.25, 0.27]:
#     thermal_factor_qp = (1 + np.exp(-h * w * 1e9 / (kB * T_qp)))
#     for idx in range(len(phi_ext)):
#         gamma_qp[idx] = r_qp_finiteT(E_l, E_c, E_j, w[idx], qp_element[idx], T_qp)*thermal_factor_qp[idx]
#     # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='r')

# for x_qp in [3e-8]:
#     Q_qp = 1.0 / x_qp
#     for idx in range(len(phi_ext)):
#         gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx], T_qp)
#     plt.semilogy(w, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--', color = 'orange')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')
# #

# plt.ylim([1e1,1e4])
# plt.yticks([5e2,1e3,2e3,5e3])
# plt.xlabel(r'$\Phi_e/\Phi_o$', fontsize = 18)
# plt.xticks([0,0.5])
# plt.ylabel(r'$T_1/Q$', fontsize = 18)

############################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Summary'
# fname='20170825_T1.txt'
# path = directory + '\\'+ fname
# data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
# current = data[:,0]
# freq = data[:,1]
# T1 = data[:,2]
# plt.errorbar(freq, T1, fmt='s', mfc='none', mew=2.0, mec='b', label="0mA")
fname = 'T1_60to62mA.txt'
path = directory + '\\'+ fname
data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
current = data[:,0]
freq = data[:,1]
T1 = data[:,2]
matrix_element_interp = np.interp(freq, w, p_element, period = 3.8)
#plt.plot((current-59.59)/1.53 * 0.5, matrix_element_interp, 's')
#plt.plot(phi_ext , p_element)
plt.errorbar(freq**alpha, T1, fmt='s', mfc='none', mew=2.0, mec='b')
plt.yscale("log")
plt.xscale("log")
# plt.errorbar(freq**alpha, T1*matrix_element_interp**2, fmt='s', mfc='none', mew=2.0, mec='b', label="60mA")
# plt.errorbar((current)/1.53 * 0.5, T1, fmt='s', mfc='none', mew=2.0, mec='blue')
# plt.errorbar((current-59.59)/1.53 * 0.5, T1, fmt='s', mfc='none', mew=2.0, mec='blue')
############################################################################
# plt.grid()
# plt.xticks([0,0.5])
plt.xticks([])
# plt.yticks([])
# plt.yticks([10,100,1000])
#plt.ylim([5,135])
# plt.xlim(0,0.51)
plt.tick_params(labelsize=18)
# plt.legend()
plt.show()