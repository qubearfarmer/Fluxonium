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
E_j = 3

phi_ext = np.linspace(-0.05,0.55,1001)
sensitivity=np.zeros(len(phi_ext))
iState = 0
fState = 1

dPhi = 0.001
dE = 0.0001
for idx,phi in enumerate(phi_ext):
    trans_energy1 = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[fState]- \
                    H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[iState]
    trans_energy2 =H(N, E_l, E_c, E_j , (phi+dPhi) * 2 * np.pi).eigenenergies()[fState] - \
                   H(N, E_l, E_c, E_j , (phi+dPhi) * 2 * np.pi).eigenenergies()[iState]
    sensitivity[idx] = (trans_energy1-trans_energy2)/dPhi

#Sensitivity unit is GHz/ (flux/flux_q)
plt.semilogy(phi_ext,(1e-5*abs(sensitivity)*1e9)**-1 *1e6)
plt.show()