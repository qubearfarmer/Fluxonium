from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from matplotlib import pyplot as plt
import numpy as np

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
#Define parameters here
N = 50
E_l = 0.7
E_c = 0.8
E_j = 2.7
level_num = 15
phi_ext = np.linspace(0,0.51,2001)
w=np.zeros(len(phi_ext))
sensitivity=np.zeros(len(phi_ext))
iState = 0
fState = 1
dPhi = 0.51/2000

for idx,phi in enumerate(phi_ext):
    trans_energy1 = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[fState]- \
                    H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[iState]
    # trans_energy2 =H(N, E_l, E_c, E_j , (phi+dPhi) * 2 * np.pi).eigenenergies()[fState] - \
    #                H(N, E_l, E_c, E_j , (phi+dPhi) * 2 * np.pi).eigenenergies()[iState]
    # sensitivity[idx] = (trans_energy2-trans_energy1)/dPhi
    w[idx] = trans_energy1
dw = np.diff(w)
dw2 = np.diff(dw)
#Sensitivity unit is GHz/ (flux/flux_q)
fig, ax1 = plt.subplots(figsize=(10, 7))
plt.tick_params(labelsize = 20.0)
ax1. plot(phi_ext, w, linewidth = 2.0, color = 'k')
ax1.set_yticks(np.linspace(0,4,5))
ax1.set_ylim([0,4])
ax2 = ax1.twinx()
plt.tick_params(labelsize = 20.0)
ax2.plot(phi_ext[0:-1],abs(dw/dPhi), '--', linewidth = 2.0, color = 'blue')
# ax2.plot(phi_ext[0:-2],abs(dw2/dPhi**2), '--', linewidth = 2.0, color = 'red')
ax2.set_xticks([0,0.5])
ax2.set_xlim([0,0.5])
ax2.set_ylim([0,16])
ax2.set_yticks([0,16])

plt.show()