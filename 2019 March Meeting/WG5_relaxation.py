import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp as r_qp
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp_array as r_qp_array

# plt.rc('text', usetex=True)
# plt.rc('font', family='sans-serif')
#Define file directory

directory = "C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results"
fname = "Relaxation_blochnium_wg5"
path = directory + "\\" + fname

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.626e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
#######################################################################################
N = 50
E_l = 0.437
E_c = 2.265
E_j = 6.487
level_num = 15
chain_num = 460
kB = 1.38e-23

iState = 0
fState = 1
phi_ext = np.linspace(0,0.5,101)
p_element = np.zeros(len(phi_ext))
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
gamma_cap = np.zeros(len(phi_ext))
gamma_cap_chain1 = np.zeros(len(phi_ext))
gamma_cap_chain2 = np.zeros(len(phi_ext))
gamma_flux = np.zeros(len(phi_ext))
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
T_diel = 1.0e-3
thermal_factor_diel = (1+np.exp(-h*w*1e9/(kB*T_diel)))
T_qp=1.0e-3
thermal_factor_qp = (1+np.exp(-h*w*1e9/(kB*T_qp)))
# C_chain = 36.0e-15
# Cg = 36.0e-18
#
for Q_cap in [3e5]:
    for idx in range(len(phi_ext)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_j, Q_cap, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
    label_text = 'Dielectric loss tangent = ' + str(Q_cap**-1)
    plt.semilogy(phi_ext, 1.0/gamma_cap *1e6, linewidth= 2.0, linestyle ='-',  label = label_text)

for x_qp in [5e-6, 20e-6]:
    Q_qp = 1.0/x_qp
    for idx in range(len(phi_ext)):
        gamma_qp[idx] = r_qp(E_l, E_c, E_j, Q_qp, w[idx], qp_element[idx])
    # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
    label_text = 'Tunnel junction quasiparticle x_qp = ' + str(x_qp)
    plt.semilogy(phi_ext, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--', label = label_text)

# for T_qp in [0.25, 0.28]:
#     thermal_factor_qp = (1 + np.exp(-h * w * 1e9 / (kB * T_qp)))
#     for idx in range(len(phi_ext)):
#         gamma_qp[idx] = r_qp_finiteT(E_l, E_c, E_j, w[idx], qp_element[idx], T_qp)*thermal_factor_qp[idx]
#     # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='r')

for x_qp in [1e-8, 20e-8]:
    Q_qp = 1.0 / x_qp
    for idx in range(len(phi_ext)):
        gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_j, Q_qp, w[idx], p_element[idx])*thermal_factor_qp[idx]
    label_text = 'Chain quasiparticle x_qp = ' + str(x_qp)
    plt.semilogy(phi_ext, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--', label = label_text)
    # plt.semilogy(w, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')


plt.xlabel(r'$\Phi/\Phi_o$')
plt.ylabel(r'$T_1 (\mu s)$')
##############################################################################################
I0 = 2.199
I_period = 2.469 * 2

FreqSingle = np.array(
    [6.6199, 4.5596, 4.2657, 3.9635, 3.6598, 3.3561, 1.8472, 1.5516, 1.2614, 0.9822, 0.7298, 0.54, 0.4864])
CurrentSingle = np.array([2.5, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.674, 4.665, 4.67, 4.675, 4.68])
T1Single = np.array([8.81, 19.3, 11.9, 29.6, 30.2, 27.3, 25.2, 23, 28, 41.6, 42.3, 48.1, 11.5, 147, 118, 102, 173])
T1ErrSingle = np.array([0.17, 0.49, 0.57, 0.66, 0.37, 0.5, 1.1, 1.9, 3.9, 4.9, 1.2, 1.2, 2.4, 35, 17, 14, 41])
T2Single = np.array([3.18, 4.85, 3.42, 3.8, 3.79, 3.75, 3.27, 4.11, 3.34, 4.29, 5.66, 5.33, 7.73])
T2ErrSingle = np.array([0.11, 0.42, 0.25, 0.26, 0.26, 0.35, 0.26, 0.48, 0.3, 0.54, 0.64, 0.27, 0.95])
FluxSingle = (CurrentSingle - I0) / I_period

FreqRepeated = np.array(
    [7.18915, 7.1005, 6.8807, 6.338, 6.0534, 5.7595, 5.462, 5.169, 4.87, 3.0522, 2.7488, 2.4463, 2.1462])
CurrentRepeated = np.array([2.22, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.7, 3.8, 3.9, 4, 4.61, 4.62, 4.63, 4.64, 4.65, 4.66])
T1Repeated = np.array([2.87, 3.74, 1.61, 7.37, 17.3, 21.2, 21.4, 17.1, 11.5, 8.98, 34.9, 17.1, 45.4, 48.7, 54.8, 57.8, 88.1, 95.3, 105])
T1ErrRepeated = np.array([0.22, 0.4, 0.068, 1.1, 1.3, 4.6, 0.78, 0.82, 0.68, 1, 2.5, 3.1, 5.3, 3.6, 3.1, 1.1, 12, 6.5, 9.1])
T2Repeated = np.array([4.34, 3.62, 2.05, 3.01, 3.77, 3.56, 3.85, 3.17, 2.97, 3.24, 3.6, 2.72, 3.46])
T2ErrRepeated = np.array([0.32, 0.41, 0.075, 0.76, 0.072, 0.17, 0.088, 0.11, 0.075, 0.4, 0.07, 0.32, 0.077])
FluxRepeated = (CurrentRepeated - I0) / I_period
plt.errorbar(FluxSingle, T1Single, yerr = T1ErrSingle, fmt = 's', mfc = 'none', mew = 1.0, mec = 'b', ecolor = 'b', label=r'$T_1$ Single')
plt.errorbar(FluxRepeated, T1Repeated, yerr = T1ErrRepeated, fmt = 'h', mfc = 'none', mew = 1.0, mec = 'b', ecolor = 'b', label=r'$T_1$ Repeated')

plt.legend()
plt.show()