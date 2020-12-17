import numpy as np
from matplotlib import pyplot as plt
from qutip import*

Na = 30
Nb = 30
E_la=1.0861245476741852
E_ca=1.0643555278248207
E_ja=4.617492828540067
E_lb=1.8806129530062636
E_cb=1.034555695897724
E_jb=5.054123738551083
J_c=0.3513666817629428
J_l = 0
level_num = 24

phi_ext_a = 0.5 * 2 * np.pi
phi_ext_b = 0.5 * 2 * np.pi

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
Hl = -J_l * phi_a*phi_b
H = Ha + Hb + Hc + Hl

eigen_energies, eigen_states = H.eigenstates()
elementa = na.matrix_element(eigen_states[0],eigen_states[2])
elementb = nb.matrix_element(eigen_states[0],eigen_states[2])
print ("From 00 to 01, A",(elementa))
print ("From 00 to 01, B",(elementb))
elementa = na.matrix_element(eigen_states[1],eigen_states[3])
elementb = nb.matrix_element(eigen_states[1],eigen_states[3])
print ("From 10 to 11, A",(elementa))
print ("From 10 to 11, B",(elementb))

elementa = na.matrix_element(eigen_states[0],eigen_states[1])
elementb = nb.matrix_element(eigen_states[0],eigen_states[1])
print ("From 00 to 10, A",(elementa))
print ("From 00 to 10, B",(elementb))
elementa = na.matrix_element(eigen_states[2],eigen_states[3])
elementb = nb.matrix_element(eigen_states[2],eigen_states[3])
print ("From 01 to 11, A",(elementa))
print ("From 01 to 11, B",(elementb))