import numpy as np
from qutip import*
from matplotlib import pyplot as plt


e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

Na = 30
Nb = 30
B_coeff = 60

E_la=1.08
E_ca=1.06
E_ja=4.6

E_lb=1.88
E_cb=1.03
E_jb=5.05

J_c = 0.35

phi_ext_a = (0.5) * 2 * np.pi
phi_ext_b = (0.5) * 2 * np.pi

a = tensor(destroy(Na), qeye(Nb))
phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)

b = tensor(qeye(Na), destroy(Nb))
phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)

ope_a = 1.0j * (phi_a - phi_ext_a)
Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
ope_b = 1.0j * (phi_b - phi_ext_b)
Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())

Hc = J_c * na * nb
H = Ha + Hb + Hc
eigen_energies, eigen_states = H.eigenstates()

print('nA, 00-10: ', str(na.matrix_element(eigen_states[0], eigen_states[1])))
print('nB, 00-10: ', str(nb.matrix_element(eigen_states[0], eigen_states[1])))
print('nA, 01-11: ', str(na.matrix_element(eigen_states[2], eigen_states[3])))
print('nB, 01-11: ', str(nb.matrix_element(eigen_states[2], eigen_states[3])))
print('nA, 00-01: ', str(na.matrix_element(eigen_states[0], eigen_states[2])))
print('nB, 00-01: ', str(nb.matrix_element(eigen_states[0], eigen_states[2])))
print('nA, 10-11: ', str(na.matrix_element(eigen_states[1], eigen_states[3])))
print('nB, 10-11: ', str(nb.matrix_element(eigen_states[1], eigen_states[3])))