import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp_array as r_qp_array

#######################################################################################
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\T1_lowField_1.txt'
freq = np.genfromtxt(path)[:,0]
t1 = np.genfromtxt(path)[:,1]
t1_err = np.genfromtxt(path)[:,2]
plt.errorbar(freq, t1, yerr=t1_err, fmt='s', mfc='none', mew=2.0, mec='b')

path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\T1_lowField_2.txt'
freq = np.genfromtxt(path)[:,0]
t1 = np.genfromtxt(path)[:,1]
t1_err = np.genfromtxt(path)[:,2]
plt.errorbar(freq, t1, yerr=t1_err, fmt='s', mfc='none', mew=2.0, mec='b')

path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\T1_lowField_3.txt'
freq = np.genfromtxt(path)[:,0]
t1 = np.genfromtxt(path)[:,1]
t1_err = np.genfromtxt(path)[:,2]
plt.errorbar(freq, t1, yerr=t1_err, fmt='s', mfc='none', mew=2.0, mec='b', label = '1.1mA')
###########################
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\T1_30mA.txt'
freq = np.genfromtxt(path)[:,0]
t1 = np.genfromtxt(path)[:,1]
t1_err = np.genfromtxt(path)[:,2]
plt.errorbar(freq, t1, yerr=t1_err, fmt='s', mfc='none', mew=2.0, mec='r', label = '30.1mA')
# plt.yscale('log')
###############################################################################################
directory = "C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results"
simulation = "Relaxation_JuliusIV"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
# plt.figure(figsize=[20,10])
#######################################################################################
N = 50
E_l=0.5825088902476563
E_c=1.0019222206424947
E_j=3.4376199557356957
level_num = 15

kB=1.38e-23

iState = 0
fState = 1
phi_ext = np.linspace(0.0,0.5,101)
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
# np.savetxt(path + '_flux.txt', phi_ext)
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
T_diel = 20.0e-3
thermal_factor_diel = (1+np.exp(-h*w*1e9/(kB*T_diel)))
T_qp=20.0e-3
thermal_factor_qp = (1+np.exp(-h*w*1e9/(kB*T_qp)))
alpha =1

for Q_cap in [0.5e6]:
    for idx in range(len(phi_ext)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
    plt.semilogy(w*1000, 1.0 / gamma_cap * 1e6, linewidth=2.0, linestyle ='--', color='blue', label='tan_diel = 2e-6')
    # plt.semilogy(w ** alpha, 1.0 / gamma_cap * 1e6 , linewidth=2.0, linestyle='--', color='orange')
    # plt.semilogy(phi_ext, 1.0/gamma_cap *1e6, linewidth= 2.0, linestyle ='--', color='orange')

for x_qp in [1e-6]:
    Q_qp = 1.0/x_qp
    for idx in range(len(phi_ext)):
        gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp, w[idx], qp_element[idx])*thermal_factor_diel[idx]
    plt.semilogy(w*1000, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--', color = 'orange',label = 'x_qp = 1e-6')
    # plt.semilogy(phi_ext, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='k')
#
# for T_qp in [0.25, 0.27]:
#     thermal_factor_qp = (1 + np.exp(-h * w * 1e9 / (kB * T_qp)))
#     for idx in range(len(phi_ext)):
#         gamma_qp[idx] = r_qp_finiteT(E_l, E_c, E_j, w[idx], qp_element[idx], T_qp)*thermal_factor_qp[idx]
#     # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='r')

for x_qp in [2e-8]:
    Q_qp = 1.0 / x_qp
    for idx in range(len(phi_ext)):
        gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx])*thermal_factor_diel[idx]
    plt.semilogy(w*1000, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--', color = 'magenta', label = 'x_qp_array = 2e-8')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')
################################################################################################
plt.xlabel('Frequency (MHz)', fontsize = 18.0)
plt.ylabel('T1(us)', fontsize = 18.0)



plt.tick_params(labelsize = 16.0)
plt.legend()
plt.show()