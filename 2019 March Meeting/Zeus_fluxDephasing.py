import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
#Define parameters here
N = 50
E_l = 0.203
E_c = 7.61
E_j = 7.287

phi_ext = np.linspace(0.0,0.5,1001)
sensitivity=np.zeros(len(phi_ext))
energy=np.zeros(len(phi_ext))
iState = 0
fState = 1

dPhi = 0.001
dE = 0.001
for idx,phi in enumerate(phi_ext):
    trans_energy1 = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[fState]- \
                    H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[iState]
    trans_energy2 =H(N, E_l, E_c, E_j , (phi+dPhi) * 2 * np.pi).eigenenergies()[fState] - \
                   H(N, E_l, E_c, E_j , (phi+dPhi) * 2 * np.pi).eigenenergies()[iState]
    sensitivity[idx] = (trans_energy2-trans_energy1)/dPhi
    energy[idx] = trans_energy1
sensitivity_check = np.diff(energy)/(0.5-0.0)*1000
# for idx,phi in enumerate(phi_ext):
#     trans_energy1 = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[fState]- \
#                     H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[iState]
#     trans_energy2 =H(N, E_l, E_c, E_j+dE , (phi) * 2 * np.pi).eigenenergies()[fState] - \
#                    H(N, E_l, E_c, E_j+dE , (phi) * 2 * np.pi).eigenenergies()[iState]
#     sensitivity[idx] = (trans_energy2-trans_energy1)/dE
#     energy[idx] = trans_energy1
#First order
# plt.plot(abs(sensitivity))
# plt.plot(abs(sensitivity_check))
#Second order
# plt.plot(np.diff(np.diff(energy))/((0.50-0.0)/1000)**2)
# plt.plot(np.diff(sensitivity)/(0.5-0)*1000)
# plt.plot(sensitivity_check)
#Sensitivity unit is GHz/ (flux/flux_q)
# fig, ax1 = plt.subplots(figsize=(10, 7))
# plt.tick_params(labelsize = 18.0)
# ax1. plot(phi_ext, energy, linewidth = 2.0, color = 'k')
# ax2 = ax1.twinx()
# plt.tick_params(labelsize = 18.0)
# ax2.plot(phi_ext,abs(sensitivity), '--', )
plt.semilogy(phi_ext,(1e-6*2*np.pi*abs(sensitivity)*1e9*np.sqrt(np.log(2)))**-1 *1e6, linestyle = '--', label =r'$A_\Phi = 1e-6 \Phi_o$')
plt.semilogy(phi_ext,(10e-6*2*np.pi*abs(sensitivity)*1e9*np.sqrt(np.log(2)))**-1 *1e6, linestyle = '--', label =r'$A_\Phi = 1e-5 \Phi_o$')

############################################################################
# directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Summary'
# fname = 'T2_62mA.txt'
# path = directory + '\\'+ fname
# data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
# current = data[:,0]
# T2 = data[:,1]
# ax2.errorbar((current-59.587)/1.53 * 0.5, T2, fmt='s', mfc='none', mew=2.0, mec='blue')
############################################################################

plt.grid()
plt.ylabel(r'$T_2(\mu s)$')
plt.xlabel(r'$\Phi / \Phi_o$')
plt.ylim([0,1000])
plt.show()