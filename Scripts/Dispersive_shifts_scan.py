from Fluxonium_hamiltonians.Single_small_junction import charge_dispersive_shift as nChi
from Fluxonium_hamiltonians.Single_small_junction import flux_dispersive_shift as pChi
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Dispersive_shift_hfq"
path = directory + "\\" + simulation
# '''
N = 50
E_c_array = np.linspace(0.5,3,26)
E_j_array = np.linspace(2,12,101)
E_l_array = np. linspace(0.1, 0.5, 5)
level_num = 50
g = 0.1
# iState = 0
# fState = 1
phi_ext = 0.5
wr = np.linspace(4,12,801)
charge_chi_01 = np.zeros(len(wr))
charge_chi_12 = np.zeros(len(wr))
charge_chi_02 = np.zeros(len(wr))
for E_l in E_l_array:
    for E_c in E_c_array:
        for E_j in E_j_array:
            path = directory + "\\" + simulation
            path = path +'_E_l='+str(E_l)+' E_c='+str(E_c)+' E_j='+str(E_j)
            for idx, w in enumerate (wr):
                charge_chi_01[idx] = nChi(N, level_num, E_l, E_c, E_j, phi_ext * 2 * np.pi, 0, 1, w, g)
                charge_chi_12[idx] = nChi(N, level_num, E_l, E_c, E_j, phi_ext * 2 * np.pi, 1, 2, w, g)
                charge_chi_02[idx] = nChi(N, level_num, E_l, E_c, E_j, phi_ext * 2 * np.pi, 0, 2, w, g)
            np.savetxt(path + '_charge_chi_01.txt', charge_chi_01)
            np.savetxt(path + '_charge_chi_12.txt', charge_chi_12)
            np.savetxt(path + '_charge_chi_02.txt', charge_chi_02)


#Plotting
# E_l = 0.1
# E_c = 2
# E_j = 8
# phi_ext = np.linspace(0,0.5,101)
# wr = np.linspace(5,25,101)
# path = directory + "\\" + simulation
# path = path+'_E_l='+str( E_l)+' E_c='+str(E_c)+' E_j='+str(E_j)
# chi = np.genfromtxt(path+'_flux.txt')
#
# fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.25, bottom=0.25)
# l, = plt.plot(phi_ext,chi[:,0],'b.')
# plt.grid()
# plt.ylim([-2,2])
# plt.text(0.01, 1.5, 'El=' + str(E_l) + ', Ec=' + str(E_c) + ', Ej=' + str(E_j))
#
# axcolor = 'lightgoldenrodyellow'
# axCav = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
# sCav= Slider(axCav, 'Cavity index', 0 , len(wr)-1, valinit = 0, valfmt='%0.00f')
# def update(cav_index):
#     cav_index = sCav.val
#     l.set_ydata(chi[:,cav_index])
#     plt.title('Cav res='+str(wr[cav_index]))
#
#
# sCav.on_changed(update)
# plt.xlabel(r'$\Phi_{ext}/Phi_o$')
# plt.ylabel(r'$\chi_{01}$')
# plt.show()
