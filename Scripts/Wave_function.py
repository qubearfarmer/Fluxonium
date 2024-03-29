import numpy as np
from matplotlib import pyplot as plt
from scipy.special import eval_hermite as hpoly

from Fluxonium.Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

plt.rc('font', family='serif')

#Define file directory
directory = "C:\\Users\\nguyen89\Documents\\Fluxonium simulation"
simulation = "Spectrum_potential_small_junction"
path = directory + "\\" + simulation
#Qubit and computation parameters
N = 100
E_l = 0.5
E_c = 0.8
E_j = 2.7
level_num = 10
####################################################################################################
phi_ext = 0.5
H = bare_hamiltonian(N,E_l,E_c,E_j, phi_ext*2*np.pi)
evalues, evectors = H.eigenstates()
# print evectors[0]
# print evectors[0].full()
# print evectors[0].full()[1,0]

def ho_wf(phi,l,Ec,El):
    ratio = (8.0*Ec/El)**(0.25)
    coeff = (2.0**l*np.math.factorial(l)*np.sqrt(np.pi)*ratio)**(-0.5)
    return coeff * np.exp(-0.5*(phi/ratio)**2) * hpoly(l,phi/ratio)

phi = np.linspace(-10,10,201)

fig, ax1 = plt.subplots(figsize=(10, 10))
ax1.tick_params(labelsize=18)
# print real(evectors[0].full()[0])[0]
for state_idx in range(0,4):
    wFunc = np.zeros(len(phi))
    for lvl_idx in range(N):
        coeff = np.real(evectors[state_idx].full()[lvl_idx, 0])
        wFunc = wFunc + coeff*ho_wf(phi, lvl_idx, E_c, E_l)
    ax1.plot(phi, wFunc+evalues[state_idx], linewidth = 2.0)

#
potential = 0.5*E_l*phi**2 - E_j*np.cos(phi-phi_ext*2*np.pi)
ax2 = ax1.twinx()
ax2.plot(phi, potential, linewidth = 5.0, color = 'black')
for idx in range(4):
    ax2.plot(phi, np.ones(len(phi))*evalues[idx], linewidth = 1.0)
ax2.tick_params(labelsize=18)
ax2.set_ylim([-2,8])
ax1.set_ylim([-2,8])
ax1.set_xlim([-10,10])
ax2.set_xlim([-10,10])
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_yticks([])
ax1.axis("off")
ax2.axis('off')
plt.show()