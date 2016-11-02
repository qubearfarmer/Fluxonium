from Fluxonium_hamiltonians.Single_small_junction import charge_dispersive_shift as nChi
from Fluxonium_hamiltonians.Single_small_junction import flux_dispersive_shift as pChi
import numpy as np
from matplotlib import pyplot as plt

N = 50
E_l = 0.5
E_c = 2.5
E_j = 10
level_num = 15
g = 0.2

iState = 0
fState = 1
phi_ext = np.linspace(0,0.5,501)
w = 10
chi = np.zeros(len(phi_ext))

for idx, phi in enumerate(phi_ext):
    chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, iState, fState, w, g)

plt.plot(phi_ext, chi, 'b.')
plt.show()