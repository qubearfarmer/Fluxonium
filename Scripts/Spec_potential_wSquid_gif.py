from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian_alt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from images2gif import writeGif
from PIL import Image
from matplotlib import rc
plt.rc('font', family='serif')
rc('text', usetex=False)

#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Spectrum_potential_wSquid"
path = directory + "\\" + simulation
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

#Qubit and computation parameters
N = 50
E_l = 0.722729827116
E_c = 0.552669197076
E_j_sum = 17.61374383
A_j = 4.76321410213e-12
A_c = 1.50075181762e-10
d = 0.125005274368
beta_squid = 0.129912406349
beta_ext = 0.356925557542
B_coeff = 60
level_num = 20
E_j1 = 0.5 * E_j_sum * (1 + d)
E_j2 = 0.5 * E_j_sum * (1 - d)
current = np.linspace(0.01,0.04,16)
phi = np.linspace(-10,10,200)
energies = np.zeros((level_num,len(current)))
potential = np.zeros((len(phi),len(current)))
####################################################################################################################################
#Simulation part
'''
for idx in range(len(current)):
    flux_squid = current[idx] * B_coeff * A_j * 1e-4
    flux_ext = current[idx] * B_coeff * A_c * 1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idy,idx] = H.eigenenergies()[idy]
    for idy in range(len(phi)):
        potential[idy,idx] = 0.5 * E_l * phi[idy] ** 2 - E_j1 * np.cos(2 * np.pi * (flux_ext / phi_o - beta_ext) - phi[idy]) \
                                                - E_j2 * np.cos(phi[idy] + 2 * np.pi * (flux_squid / phi_o - beta_squid) - 2 * np.pi * (flux_ext / phi_o - beta_ext))
np.savetxt(path+'_energy.txt', energies)
np.savetxt(path+'_potential.txt', potential)
'''
####################################################################################################################################
#Plotting from file
energies_f = np.genfromtxt(path+'_energy.txt')
potential_f = np.genfromtxt(path+'_potential.txt')
####################################################################################################################################
#Save all the lots for gif
file_names = []
for idx in range(len(current)):
    flux_squid = current[idx] * B_coeff * A_j * 1e-4
    flux_ext = current[idx] * B_coeff * A_c * 1e-4
    fig=plt.figure(figsize=[10,10])
    plt.ylim([-20, 40])
    plt.yticks([-15,0,15,30,45])
    plt.xlim([phi[0], phi[-1]])
    # plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.tick_params(labelsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.plot(phi, potential_f[:, idx], color='k', linewidth='5')
    energy = np.zeros(len(phi))
    for idy in range(level_num):
        energy[:] = energies_f[idy, idx]
        plt.plot(phi,energy)
    # ax.set_title('Current='+str(current[idx]*1e3)+'mA', fontsize=26)
    plt.title(r'$\phi_1 + \phi_2=$' + str(np.round((flux_squid+flux_ext)/phi_o,3)), fontsize=32, y=1.05)
    fn = 'C:\Data\Fluxonium #10 simulations' + '\\gif\\'+ 'img'+str(idx)+'.png'
    file_names = np.append(file_names, fn)
    fig.savefig(fn)
    plt.close("all")
######################################################################################################
#Widgets
# fig, ax = plt.subplots(figsize=[12,12])
# plt.ylim([-20,40])
# plt.subplots_adjust(left=0.1, bottom=0.1)
# #potential
# l, = plt.plot(phi,potential_f[:,0], color = 'k', linewidth = '4')
# # plt.ylim([-5,10])
# #energy
# energy = np.zeros(len(phi))
# d = {}
# for idx in range(5):
#     energy[:] = energies_f[idx,0]
#     d["m{0}".format(idx)], = plt.plot(phi, energy)
#
# ax.set_xlabel(r'$\varphi$')
# ax.set_ylabel('Energy')
#
# #Slider defined here
# axcolor = 'lightgoldenrodyellow'
# axFlux = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
# sFlux = Slider(axFlux, 'Current index', 0 , len(current)-1, valinit = 0)#, valfmt='%0.00f')
# energy = np.zeros((len(phi),level_num))
# def update(flux_index):
#     flux_index = sFlux.val
#     l.set_ydata(potential_f[:,flux_index])
#     plt.title("Current" +str(sFlux.val)+"="+str(current[flux_index]*1e3)+"mA")
#     for idx in range(level_num):
#         energy[:,idx] = energies_f[idx, flux_index]
#         d["m{0}".format(idx)].set_ydata(energy[:,idx])
#
# sFlux.on_changed(update)
# plt.show()

