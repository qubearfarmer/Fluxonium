import numpy as np
from matplotlib import pyplot as plt
from qutip import*

e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

Na = 20
Nb = 20


E_la=1.
E_ca=1.
E_ja=4.

E_lb=1.
E_cb=1.
E_jb=3.8
phi_ext = 0.5*2*np.pi

Jc = np.linspace(0,1,101)
na_element = np.zeros(len(Jc))
na2_element = np.zeros(len(Jc))
energies = np.zeros((len(Jc), 4))
a = tensor(destroy(Na), qeye(Nb))
phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)

b = tensor(qeye(Na), destroy(Nb))
phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)

ope_a = 1.0j * (phi_a - phi_ext)
Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
ope_b = 1.0j * (phi_b - phi_ext)
Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())


for idx in range(len(Jc)):
    Hc = Jc[idx] * na * nb
    H = Ha + Hb + Hc
    eigen_energies, eigen_states = H.eigenstates()
    na_element[idx] = abs(na.matrix_element(eigen_states[1], eigen_states[2]))
    na2_element[idx] = abs((na*na).matrix_element(eigen_states[1], eigen_states[2]))

    for idy in range(4):
        energies[idx, idy] = eigen_energies[idy]

plt.figure(1)
for idx in range(4):
    plt.plot(Jc, energies[:,idx])
plt.xlabel('Jc (GHz)', size = 18.0)
plt.ylabel('Eigen energy (GHz)', size = 18.0)
plt.tick_params(labelsize = 14.0)

plt.figure(2)
plt.plot(Jc, na_element, label = 'Direct drive')
plt.plot(Jc, na2_element, label = '2 photon drive')
plt.xlabel('Jc (GHz)', size = 18.0)
plt.ylabel('Matrix element', size = 18.0)
plt.tick_params(labelsize = 14.0)
plt.legend()
plt.show()