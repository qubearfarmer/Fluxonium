from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Spectrum_potential_small_junction"
path = directory + "\\" + simulation
#Qubit and computation parameters
N = 50
E_l = 0.1
E_c = 12
E_j = 10

phi_ext = np.linspace(0,0.5,501)
phi = np.linspace(-10,10,201)
level_num = 10
energies = np.zeros((level_num, len(phi_ext)))
potential = np.zeros((len(phi), len(phi_ext)))
####################################################################################################
#Simulation part
# '''
for idx in range(len(phi_ext)):
    H = bare_hamiltonian(N, E_l, E_c, E_j, phi_ext[idx]*2*np.pi)
    for idy in range(level_num):
        energies[idy,idx] = H.eigenenergies()[idy]
    for idy in range(len(phi)):
        potential[idy,idx] = 0.5*E_l*phi[idy]**2 - E_j*np.cos(phi[idy]-phi_ext[idx]*2*np.pi)

np.savetxt(path+'_energy.txt', energies)
np.savetxt(path+'_potential.txt', potential)
# '''
#####################################################################################################
#Plotting from file
energies_f = np.genfromtxt(path+'_energy.txt')
potential_f = np.genfromtxt(path+'_potential.txt')

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
#potential
l, = plt.plot(phi,potential_f[:,0], color = 'k', linewidth = 2.0)
#energy
energy = np.zeros(len(phi))
d = {}
for idx in range(level_num):
    energy[:] = energies_f[idx,0]
    d["m{0}".format(idx)], = plt.plot(phi, energy)

ax.set_xlabel(r'$\varphi/2\pi$')
ax.set_ylabel('Energy')

#Slider defined here
axcolor = 'lightgoldenrodyellow'
axFlux = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
sFlux = Slider(axFlux, 'Flux index', 0 , len(phi_ext)-1, valinit = 0, valfmt='%0.00f')
energy = np.zeros((len(phi),level_num))
def update(flux_index):
    flux_index = sFlux.val
    l.set_ydata(potential_f[:,flux_index])
    plt.title("Normalized flux="+str(phi_ext[flux_index]))
    for idx in range(level_num):
        energy[:,idx] = energies_f[idx, flux_index]
        d["m{0}".format(idx)].set_ydata(energy[:,idx])


sFlux.on_changed(update)

plt.show()
