from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian_alt
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import eval_hermite as hpoly

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
phi = np.linspace(-10,10,201)
energies = np.zeros(level_num)
potential = np.zeros(len(phi))

current = 0.039777
flux_squid = current * B_coeff * A_j * 1e-4
flux_ext = current * B_coeff * A_c * 1e-4
H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
evalues, evectors = H.eigenstates()

def ho_wf(phi,l,Ec,El):
    ratio = (8.0*Ec/El)**(0.25)
    coeff = (2.0**l*np.math.factorial(l)*np.sqrt(np.pi)*ratio)**(-0.5)
    return coeff * np.exp(-0.5*(phi/ratio)**2) * hpoly(l,phi/ratio)

phi = np.linspace(-10,10,201)

fig, ax1 = plt.subplots(figsize=(10, 10))
ax1.tick_params(labelsize=18)
# print real(evectors[0].full()[0])[0]
for state_idx in range(4):
    wFunc = np.zeros(len(phi))
    for lvl_idx in range(N):
        coeff = np.real(evectors[state_idx].full()[lvl_idx, 0])
        wFunc = wFunc + coeff*ho_wf(phi, lvl_idx, E_c, E_l)
    ax1.plot(phi, (wFunc), linewidth = 2.0)
#
potential = 0.5 * E_l * phi ** 2 - E_j1 * np.cos(2 * np.pi * (flux_ext / phi_o - beta_ext) - phi) \
                                                - E_j2 * np.cos(phi + 2 * np.pi * (flux_squid / phi_o - beta_squid) - 2 * np.pi * (flux_ext / phi_o - beta_ext))

ax2 = ax1.twinx()
ax2.plot(phi, potential, linewidth = 3.0, color = 'black')
# for idx in range(4):
#     ax2.plot(phi, np.ones(len(phi))*evalues[idx], linewidth = 3.0)
ax2.tick_params(labelsize=18)
ax2.set_ylim([-10,30])
ax1.set_ylim([-1,1])
plt.show()