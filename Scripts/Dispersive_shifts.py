from Fluxonium_hamiltonians.Single_small_junction import charge_dispersive_shift as nChi
from Fluxonium_hamiltonians.Single_small_junction import flux_dispersive_shift as pChi
import numpy as np
from matplotlib import pyplot as plt

N = 50
E_l = 0.5
E_c = 0.8
E_j = 1.577
level_num = 20
g = 0.1

iState = 0
fState = 2

#plot dispersive shift as a function of flux
phi_ext = np.linspace(0,0.5,501)
w = 7.358
chi = np.zeros(len(phi_ext))
chi1 = np.zeros(len(phi_ext))
chi2 = np.zeros(len(phi_ext))
kappa = 50 #MHz
for idx, phi in enumerate(phi_ext):
    chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, iState, fState, w, g)
    chi1[idx] = nChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, 0, 2, w, g)
    chi2[idx] = nChi(N, level_num, E_l, E_c, E_j, phi * 2 * np.pi, 1, 2, w, g)

# chi_angle = chi*1e3/(kappa/2) *180/np.pi
plt.plot(phi_ext, chi*1e3 , 'b.')
# chi_angle = chi1*1e3/(kappa/2) *180/np.pi
# plt.plot(phi_ext, chi_angle , 'r.')
# chi_angle = chi2*1e3/(kappa/2) *180/np.pi
# plt.plot(phi_ext, chi_angle , 'g.')

#plot dispersive shift as a function of cavity frequency
# phi_ext = 0.5
# w = np.linspace(5,20,1501)
# chi = np.zeros(len(w))
# kappa = 50 #MHz
#
# iState = 0
# fState = 1
# for idx, freq in enumerate(w):
#     chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi_ext*2*np.pi, iState, fState, freq, g)
#
# chi_angle = chi*1e3/(kappa/2) *180/np.pi
# plt.plot(w, chi_angle, '.')
#
# iState = 0
# fState = 2
# for idx, freq in enumerate(w):
#     chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi_ext*2*np.pi, iState, fState, freq, g)
#
# chi_angle = chi*1e3/(kappa/2) *180/np.pi
# plt.plot(w, chi_angle, '.')
#
# iState = 1
# fState = 2
# for idx, freq in enumerate(w):
#     chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi_ext*2*np.pi, iState, fState, freq, g)
#
# chi_angle = chi*1e3/(kappa/2) *180/np.pi
# plt.plot(w, chi_angle, '.')
#
# plt.ylim([-10,10])
plt.grid()
plt.show()