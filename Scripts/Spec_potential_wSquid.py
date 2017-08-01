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
plt.figure(figsize=[6,10])
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

phi = np.linspace(-10,10,201)
energies = np.zeros(level_num)
potential = np.zeros(len(phi))

current = 0.039777
flux_squid = current * B_coeff * A_j * 1e-4
flux_ext = current * B_coeff * A_c * 1e-4
H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
for idx in range(level_num):
    energies[idx] = H.eigenenergies()[idx]
for idx in range(len(phi)):
    E_j1 = 0.5 * E_j_sum * (1 + d)
    E_j2 = 0.5 * E_j_sum * (1 - d)
    potential[idx] = 0.5 * E_l * phi[idx] ** 2 - E_j1 * np.cos(2 * np.pi * (flux_ext / phi_o - beta_ext) - phi[idx]) \
                                                - E_j2 * np.cos(phi[idx] + 2 * np.pi * (flux_squid / phi_o - beta_squid) - 2 * np.pi * (flux_ext / phi_o - beta_ext))

plt.plot(phi, potential, linewidth = 3.0, color = 'black')
for idx in range(5):
    plt.plot(phi,np.ones(len(phi))*energies[idx])
plt.xticks([])
plt.yticks([])

# directory = 'C:\\Users\\nguyen89\\Box Sync\Research\Paper Images'
# fname = 'Potential_38.142mA.eps'
# path = directory + '\\' + fname
# plt.savefig(path, format='eps', dpi=1000)

plt.show()