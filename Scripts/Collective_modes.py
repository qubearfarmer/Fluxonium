import numpy as np
from matplotlib import pyplot as plt

#Parameters here
#Ec in GHz = Ec*1.5e24
e = 1.602e-19
junc_num = 100
mode_num = 1
g_chain = np.zeros(mode_num)
C_g_a = 26e-15#F, array junction ground capacitance
C_a = 36e-15#F, array junction capacitance, using 45fF/um^2
E_g_a = 70#GHz, array junction ground charging energy
E_j_a = 11#GHz, array junction josephson energy
E_c_a = e**2 / (2*C_a)*1.5e24#GHz, array junction charging energy
E_c_b = 0.835#GHz, qubit junction charging energy

#Secondary parameters
lamb = (junc_num-1)*E_t/E_g_a
E_c_phi = ((E_c_b)**-1 + (junc_num*E_c_a)**-1 + (4.0*E_t)**-1*(1.0-2.0/3.0 *(junc_num + 1) / junc_num* lamb))**-1
E_c_e =

for i in range(1, mode_num):
    g_chain[i] = 4.0/np.sqrt(junc_num) * (E_c_phi*E_c_e/E_g_a) * np.cos(np.pi*i/junc_num)/ np.sin(np.pi*i/junc_num)**2