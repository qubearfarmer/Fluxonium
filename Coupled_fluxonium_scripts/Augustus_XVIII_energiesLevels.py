import numpy as np
from qutip import*
from matplotlib import pyplot as plt
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

Na = 25
Nb = 25
B_coeff = 60

E_la=0.45170477438306156
E_ca=0.9706755677649527
E_ja=5.842362715088368

E_lb=0.7175559802254586
E_cb=0.9963875250852217
E_jb=5.882212077372602

J_c = 0.15
phi_ext = np.linspace(0.45,0.55,101)
level_num = 20
energies = np.zeros((len(phi_ext), level_num))
#Bare levels
for idx, phi in enumerate(phi_ext):
    H = bare_hamiltonian(Na, E_la, E_ca, E_ja, phi*2*np.pi)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]
plt.plot(phi_ext, energies[:,1]-energies[:,0], color='k', linestyle = '-', label = 'bare 00-10')
plt.plot(phi_ext, energies[:,2]-energies[:,0], color='k', linestyle = '-.', label = 'bare 00-20')
plt.plot(phi_ext, energies[:,2]-energies[:,1], color='k', linestyle = '--', label = 'bare 10-20 & 11-21')

for idx, phi in enumerate(phi_ext):
    H = bare_hamiltonian(Nb, E_lb, E_cb, E_jb, phi*2*np.pi)
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]
plt.plot(phi_ext, energies[:, 1] - energies[:, 0], color='b', linestyle='-', label = 'bare 00-01')
plt.plot(phi_ext, energies[:, 2] - energies[:, 0], color='b', linestyle='-.', label = 'bare 00-02')
plt.plot(phi_ext, energies[:, 2] - energies[:, 1], color='b', linestyle='--', label = 'bare 01-02 & 11-12')

#coupled levels
# a = tensor(destroy(Na), qeye(Nb))
# phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
# na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)
#
# b = tensor(qeye(Na), destroy(Nb))
# phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
# nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)
#
# for idx, phi in enumerate(phi_ext):
#     phi_ext_a = phi * 2 * np.pi
#     phi_ext_b = phi * 2 * np.pi
#     ope_a = 1.0j * (phi_a - phi_ext_a)
#     Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
#     ope_b = 1.0j * (phi_b - phi_ext_b)
#     Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
#     Hc = J_c * na * nb
#     H = Ha + Hb + Hc
#     for idy in range(level_num):
#         energies[idx,idy] = H.eigenenergies()[idy]
# directory = 'C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results'
# fname = "Coupled_fluxonium_spectrum_AugustusXVIII_levels.txt"
# path = directory + '\\' + fname
# np.savetxt(path, energies)
##################################################################################
directory = 'C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_AugustusXVIII_levels.txt"
path = directory + '\\' + fname
energies =  np.genfromtxt(path)
plt.plot(phi_ext, energies[:,1] - energies[:,0], label = '00-10')
plt.plot(phi_ext, energies[:,2] - energies[:,0], label = '00-01')
plt.plot(phi_ext, energies[:,5] - energies[:,3], label = '11-12')
plt.plot(phi_ext, energies[:,4] - energies[:,2], label = '01-02')
plt.plot(phi_ext, energies[:,7] - energies[:,3], label = '11-21')
plt.plot(phi_ext, energies[:,6] - energies[:,1], label = '10-20')

plt.plot(phi_ext, energies[:,4] - energies[:,0])
plt.plot(phi_ext, energies[:,5] - energies[:,0])
plt.plot(phi_ext, energies[:,6] - energies[:,0])
plt.plot(phi_ext, energies[:,7] - energies[:,0])
plt.plot(phi_ext, energies[:,4] - energies[:,1])
plt.plot(phi_ext, energies[:,5] - energies[:,1])
plt.plot(phi_ext, energies[:,6] - energies[:,1])
plt.plot(phi_ext, energies[:,7] - energies[:,1])
plt.plot(phi_ext, energies[:,4] - energies[:,2])
plt.plot(phi_ext, energies[:,5] - energies[:,2])
plt.plot(phi_ext, energies[:,6] - energies[:,2])
plt.plot(phi_ext, energies[:,7] - energies[:,2])

plt.legend()
plt.show()