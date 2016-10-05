from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian_alt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

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
E_l = 0.746959655208
E_c = 0.547943694372
E_j_sum = 21.9627179709
level_num = 10
B_coeff = 60
A_j = 3.80888914574e-12
A_c = 1.49982268962e-10
beta_squid = 0.00378012644185
beta_ext = 0.341308382441
d=0.0996032153487

E_j1 = 0.5 * E_j_sum * (1 + d)
E_j2 = 0.5 * E_j_sum * (1 - d)
current = np.linspace(0,0.05,5001)
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

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
#potential
l, = plt.plot(phi,potential_f[:,0], color = 'k', linewidth = '2')
# plt.ylim([-5,10])
#energy
energy = np.zeros(len(phi))
d = {}
for idx in range(level_num):
    energy[:] = energies_f[idx,0]
    d["m{0}".format(idx)], = plt.plot(phi, energy)

ax.set_xlabel(r'$\varphi$')
ax.set_ylabel('Energy')

#Slider defined here
axcolor = 'lightgoldenrodyellow'
axFlux = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
sFlux = Slider(axFlux, 'Current index', 0 , len(current)-1, valinit = 0, valfmt='%0.00f')
energy = np.zeros((len(phi),level_num))
def update(flux_index):
    flux_index = sFlux.val
    l.set_ydata(potential_f[:,flux_index])
    plt.title("Current="+str(current[flux_index]*1e3)+"mA")
    for idx in range(level_num):
        energy[:,idx] = energies_f[idx, flux_index]
        d["m{0}".format(idx)].set_ydata(energy[:,idx])

sFlux.on_changed(update)

plt.show()
