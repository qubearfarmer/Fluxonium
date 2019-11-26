import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
#Define parameters here
N = 50
# E_l = 1.0168
# E_c = 0.835
# E_j = 2.996

E_l = 1.01
E_c = 0.835
E_j = 2.99

phi_ext = np.linspace(.45,0.51,101)
sensitivity=np.zeros(len(phi_ext))
energy=np.zeros(len(phi_ext))
iState = 0
fState = 1

dPhi = 0.0001
dE = 0.0001
for idx,phi in enumerate(phi_ext):
    trans_energy1 = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[fState]- \
                    H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[iState]
    trans_energy2 =H(N, E_l, E_c, E_j , (phi+dPhi) * 2 * np.pi).eigenenergies()[fState] - \
                   H(N, E_l, E_c, E_j , (phi+dPhi) * 2 * np.pi).eigenenergies()[iState]
    sensitivity[idx] = (trans_energy2-trans_energy1)/dPhi
    energy[idx] = trans_energy1

#Sensitivity unit is GHz/ (flux/flux_q)
fig, ax1 = plt.subplots(figsize=(7, 7))
plt.tick_params(labelsize = 18.0)
ax2 = ax1.twinx()
ax2. plot(phi_ext, energy, linewidth = 1.0, color = 'k', linestyle = '--')
plt.tick_params(labelsize = 18.0)
# ax2.plot(phi_ext,abs(sensitivity), '--', )
ax1.semilogy(phi_ext,(1.8e-6*2*np.pi*abs(sensitivity)*1e9*np.sqrt(np.log(2)))**-1 *1e6, color = 'black', linestyle = '-')
# ax2.semilogy(phi_ext,(2*1e-5*abs(sensitivity)*1e9*np.sqrt(np.log(2)))**-1 *1e6, '--', )
############################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Summary'
fname = 'T2_62mA.txt'
path = directory + '\\'+ fname
data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
current = data[:,0]
T2 = data[:,1]
T2_error = data[:,2]
ax1.errorbar((current-59.585)/1.53 * 0.5, T2, yerr = T2_error, fmt='h', mfc='none', mew=3.0, mec='g', ecolor='g')

directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Summary'
fname = 'T1_60to62mA.txt'
path = directory + '\\'+ fname
data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
current = data[:,0]
T1 = data[:,2]
# ax1.errorbar((current-59.585)/1.53 * 0.5, T1, fmt='s', mfc='none', mew=2.0, mec='b', ecolor='b')
############################################################################
ax2.set_xlim([0.45,0.51])
ax2.set_yticks([0.8,1.0,1.2,1.4])
ax1.set_ylim([5,200])
path_save = "C:\\Users\\nguyen89\\Desktop\\Paper figures 2019_11_24\Fig4a.pdf"
plt.savefig(path_save,dpi=300,format='pdf')
plt.show()