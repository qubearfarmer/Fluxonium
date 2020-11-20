import numpy as np
from matplotlib import pyplot as plt

from Fluxonium.Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
#Define parameters here
N = 50
E_l = 0.79
E_c = 1.0
E_j = 4.45

phi_ext = np.linspace(0.495,0.505,101)
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
sensitivity_check = np.diff(energy)/(0.505-0.495)*100
# for idx,phi in enumerate(phi_ext):
#     trans_energy1 = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[fState]- \
#                     H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[iState]
#     trans_energy2 =H(N, E_l, E_c, E_j+dE , (phi) * 2 * np.pi).eigenenergies()[fState] - \
#                    H(N, E_l, E_c, E_j+dE , (phi) * 2 * np.pi).eigenenergies()[iState]
#     sensitivity[idx] = (trans_energy2-trans_energy1)/dE
#     energy[idx] = trans_energy1
#First order
plt.plot(abs(sensitivity))
plt.plot(abs(sensitivity_check))
plt.plot(np.gradient(energy)/(0.505-0.495)*100)
#Second order
# plt.plot(np.diff(np.diff(energy))/((0.505-0.495)/100)**2)
# plt.plot(np.diff(sensitivity)/(0.505-0.495)*100)
# plt.plot(sensitivity_check)
#Sensitivity unit is GHz/ (flux/flux_q)
# fig, ax1 = plt.subplots(figsize=(10, 7))
# plt.tick_params(labelsize = 18.0)
# ax1. plot(phi_ext, energy, linewidth = 2.0, color = 'k')
# ax2 = ax1.twinx()
# plt.tick_params(labelsize = 18.0)
# ax2.plot(phi_ext,abs(sensitivity), '--', )
# ax2.semilogy(phi_ext[0:-1],(5*1e-6*abs(sensitivity)*1e9)**-1 *1e6, '--', )
# ax2.semilogy(phi_ext,(15*1e-6*abs(sensitivity)*1e9)**-1 *1e6, '--', )
############################################################################
# directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Summary'
# fname = 'T2_62mA.txt'
# path = directory + '\\'+ fname
# data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
# current = data[:,0]
# T2 = data[:,1]
# ax2.errorbar((current-59.587)/1.53 * 0.5, T2, fmt='s', mfc='none', mew=2.0, mec='blue')
############################################################################

# plt.grid()
plt.show()