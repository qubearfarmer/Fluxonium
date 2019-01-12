import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import charge_dispersive_shift as nChi

directory = "C:\\Users\\nguyen89\Documents\\Fluxonium simulation"
simulation = "Dispersive_shifts_scan"
path = directory + "\\" + simulation
# '''
N = 50
# E_c_array = np.linspace(0.5,3,26)
# E_j_array = np.linspace(2,12,101)
# E_l_array = np. linspace(0.1, 0.5, 5)
level_num = 50
g = 0.1
# iState = 0
# fState = 1
phi_ext = 0.5
wr = np.linspace(3,10,101)
charge_chi_01 = np.zeros(len(wr))
# charge_chi_12 = np.zeros(len(wr))
# charge_chi_02 = np.zeros(len(wr))
# for E_l in E_l_array:
#     for E_c in E_c_array:
#         for E_j in E_j_array:
#             path = directory + "\\" + simulation
#             path = path +'_E_l='+str(E_l)+' E_c='+str(E_c)+' E_j='+str(E_j)
#             for idx, w in enumerate (wr):
#                 charge_chi_01[idx] = nChi(N, level_num, E_l, E_c, E_j, phi_ext * 2 * np.pi, 0, 1, w, g)
#                 charge_chi_12[idx] = nChi(N, level_num, E_l, E_c, E_j, phi_ext * 2 * np.pi, 1, 2, w, g)
#                 charge_chi_02[idx] = nChi(N, level_num, E_l, E_c, E_j, phi_ext * 2 * np.pi, 0, 2, w, g)
#             np.savetxt(path + '_charge_chi_01.txt', charge_chi_01)
#             np.savetxt(path + '_charge_chi_12.txt', charge_chi_12)
#             np.savetxt(path + '_charge_chi_02.txt', charge_chi_02)


#Plotting
E_l = 0.7
E_c = 1
E_j = 4
phi_ext = 0.5
for idx in range(len(wr)):
    charge_chi_01[idx] = nChi(N, level_num, E_l, E_c, E_j, phi_ext * 2 * np.pi, 0, 1, wr[idx], g)

plt.plot(wr, charge_chi_01*1e3)
plt.xlabel("Cavity freq (GHz)")
plt.ylabel("$\chi$ (MHz)")
plt.show()
