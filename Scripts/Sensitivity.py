from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
from matplotlib import pyplot as plt
import numpy as np

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
#Define parameters here
N=50
E_l=0.5
E_c=2.5
E_j=10

phi_ext = np.linspace(0,0.5,101)
sensitivity=np.zeros(len(phi_ext))
iState = 0
fState = 1

dPhi = phi_o*0.1
dE = 0.0001
for idx,phi in enumerate(phi_ext):
    trans_energy1 = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[fState]- \
                    H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[iState]
    trans_energy2 =H(N, E_l+ dE, E_c, E_j , phi * 2 * np.pi).eigenenergies()[fState] - \
                   H(N, E_l+ dE, E_c, E_j , phi * 2 * np.pi).eigenenergies()[iState]
    sensitivity[idx] = (trans_energy1-trans_energy2)/dE

plt.plot(phi_ext,abs(sensitivity))
plt.show()