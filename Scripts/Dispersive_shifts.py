from Fluxonium_hamiltonians.Single_small_junction import charge_dispersive_shift as nChi
from Fluxonium_hamiltonians.Single_small_junction import flux_dispersive_shift as pChi
import numpy as np
from matplotlib import pyplot as plt

N = 50
E_l = 0.2
E_c = 1.2
E_j = 3.0

level_num = 30
g = 0.09

iState = 0
fState = 1

#plot dispersive shift as a function of flux
phi_ext = np.linspace(0.0,0.5,501)
w = 7.35
chi = np.zeros(len(phi_ext))
chi1 = np.zeros(len(phi_ext))
chi2 = np.zeros(len(phi_ext))
chi3 = np.zeros(len(phi_ext))
kappa = 5 #MHz
for idx, phi in enumerate(phi_ext):
    chi[idx]= pChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, iState, fState, w, g)
    # chi1[idx] = pChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, 2, 0, w, g)
    # chi2[idx] = nChi(N, level_num, E_l, E_c, E_j, phi * 2 * np.pi, 2, 0, w, g)
    # chi3[idx] = nChi(N, level_num, E_l, E_c, E_j, phi * 2 * np.pi, 3, 0, w, g)
#chi is in GHz
#chi_angle is in degree

# chi_angle = chi*1e3/(kappa/2) *180/np.pi
# chi1_angle = chi1*1e3/(kappa/2) *180/np.pi
# chi2_angle = chi2*1e3/(kappa/2) *180/np.pi
# chi3_angle = chi3*1e3/(kappa/2) *180/np.pi

plt.figure(1)
plt.plot(phi_ext, chi*1e3 , 'k-')
# plt.plot(phi_ext, chi1*1e3 , 'b-')
# plt.plot(phi_ext, chi2*1e3 , 'r-')
# plt.plot(phi_ext, chi3*1e3 , 'g-')
# plt.grid()
# plt.ylim([-2,2])
# plt.tick_params(labelsize = 18.0)

# plt.figure(2)
# plt.plot(phi_ext, chi_angle , 'k-')
# plt.plot(phi_ext, chi1_angle , 'b-')
# plt.plot(phi_ext, chi2_angle , 'r-')
# plt.plot(phi_ext, chi3_angle , 'g-')
# plt.ylim([-20,20])
# plt.tick_params(labelsize = 18.0)
# chi_angle = chi1*1e3/(kappa/2) *180/np.pi
# plt.plot(phi_ext, chi_angle , 'r.')
# chi_angle = chi2*1e3/(kappa/2) *180/np.pi
# plt.plot(phi_ext, chi_angle , 'g.')

# plot dispersive shift as a function of cavity frequency
# phi_ext = 0.5
# w = np.linspace(5,12,701)
# chi = np.zeros(len(w))
# kappa = 5 #MHz
# #
# iState = 0
# fState = 1
# for idx, freq in enumerate(w):
#     chi[idx]= pChi(N, level_num, E_l, E_c, E_j, phi_ext*2*np.pi, iState, fState, freq, g)

# chi_angle = chi*1e3/(kappa/2) *180/np.pi
# plt.plot(w, chi*1e3, '.')
#
# iState = 0
# fState = 2
# for idx, freq in enumerate(w):
#     chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi_ext*2*np.pi, iState, fState, freq, g)
#
# chi_angle = chi*1e3/(kappa/2) *180/np.pi
# plt.plot(w, chi*1e3, '-')
# plt.ylim([-2,2])
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