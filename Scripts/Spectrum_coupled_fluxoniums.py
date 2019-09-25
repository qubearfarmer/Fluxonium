import numpy as np
from matplotlib import pyplot as plt
from qutip import*

# directory = 'C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results'
# fname = "Coupled_fluxonium_spectrum_AugustusXVI_fit_20190906.txt"
# path = directory + '\\' + fname
#
# energies = np.genfromtxt(path)
# level_num = len(energies[0,:])
# current = np.linspace(0.5,2,751)*1e-3
# for idx in range(level_num):
#     # print(len(energies[:, idx]))
#     plt.plot(current*1e3, energies[:,idx], linestyle ='-', alpha = 0.5)
#
# plt.show()

Na = 25
Nb = 25
E_la=1.0861245476741852
E_ca=1.0643555278248207
E_ja=4.617492828540067
E_lb=1.8806129530062636
E_cb=1.034555695897724
E_jb=5.054123738551083
J_c = 0.35
J_l_array = np.linspace(0,1,51)
level_num = 24
energies = np.zeros((len(J_l_array), level_num))
################################################
phi_ext_a = 0.5 * 2 * np.pi
phi_ext_b = 0.5 * 2 * np.pi

a = tensor(destroy(Na), qeye(Nb))
phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)

b = tensor(qeye(Na), destroy(Nb))
phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)

for idx, J_l in enumerate(J_l_array):
    ope_a = 1.0j * (phi_a - phi_ext_a)
    Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
    ope_b = 1.0j * (phi_b - phi_ext_b)
    Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
    Hc = J_c * na * nb
    Hl = J_l * phi_a * phi_b
    H = Ha + Hb + Hc + Hl
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]
directory = "C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results"
simulation = "Spectrum_coupled_coupled_1"
path = directory + "\\" + simulation
np.savetxt(path + '_energies.txt', energies)
################################################
energies = np.genfromtxt(path+'_energies.txt')
for idx in range (level_num):
    plt.plot(J_l_array, energies[:,idx])
# plt.plot(J_l_array, (energies[:,1]-energies[:,0]))
# plt.plot(J_l_array, energies[:,3]-energies[:,1])
plt.xlabel('Jc(GHz)')
plt.ylabel('Energy levels (GHz)')
plt.show()
