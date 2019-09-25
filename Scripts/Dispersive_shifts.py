import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import charge_dispersive_shift as nChi

N = 50
E_l=0.6455327197108344
E_c=1.0287998197688306
E_j=3.1948994748937576
level_num = 30
g = 0.1

iState = 0
fState = 1

#plot dispersive shift as a function of flux
phi_ext = np.linspace(0.4,0.6,201)
w = 6.22
chi = np.zeros(len(phi_ext))
chi1 = np.zeros(len(phi_ext))
chi2 = np.zeros(len(phi_ext))
chi3 = np.zeros(len(phi_ext))
kappa = 4 #MHz
for idx, phi in enumerate(phi_ext):
    chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, 0, 1, w, g)
    # chi1[idx] = nChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, 2, 0, w, g)
    # chi2[idx] = nChi(N, level_num, E_l, E_c, E_j, phi * 2 * np.pi, 2, 0, w, g)
    # chi3[idx] = nChi(N, level_num, E_l, E_c, E_j, phi * 2 * np.pi, 3, 0, w, g)
#chi is in GHz
#chi_angle is in degree

# chi_angle = chi*1e3/(kappa/2) *180/np.pi
# chi1_angle = chi1*1e3/(kappa/2) *180/np.pi
# chi2_angle = chi2*1e3/(kappa/2) *180/np.pi
# chi3_angle = chi3*1e3/(kappa/2) *180/np.pi

plt.figure(1)
plt.plot(phi_ext, chi*1e3 , 'k-', label = 'A')
# plt.plot(phi_ext, chi1*1e3 , 'b-', label = '0-2')
# plt.plot(phi_ext, chi2*1e3 , 'r-')
# plt.plot(phi_ext, chi3*1e3 , 'g-', label = '0-3')
# plt.grid()
# plt.ylim([-2,2])
# plt.tick_params(labelsize = 18.0)
E_l=0.8751435686148983
E_c=1.0690006772511425
E_j=2.9378419957349036
level_num = 30
g = 0.1
for idx, phi in enumerate(phi_ext):
    chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, 0, 1, w, g)
plt.plot(phi_ext, chi*1e3 , 'b-', label = 'B')
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
# w = np.linspace(3,12,901)
# chi = np.zeros(len(w))
# kappa = 5 #MHz
# #
# iState = 1
# fState = 2
# for idx, freq in enumerate(w):
#     chi[idx]= nChi(N, level_num, E_l, E_c, E_j, phi_ext*2*np.pi, iState, fState, freq, g)

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
plt.legend()
plt.xlabel('Flux/Flux_Q')
plt.ylabel('Chi (MHz)')
plt.grid()
plt.show()