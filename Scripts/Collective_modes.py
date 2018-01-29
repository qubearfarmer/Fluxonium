import numpy as np
from matplotlib import pyplot as plt
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

#Parameters here
#Ec in GHz = Ec*1.5e24
e = 1.602e-19
N = 50
level_num = 20
junc_num = 100
mode_num = 50
phi_ext = 0.5
w_r = 7.36#GHz
kappa_r = 5e-3 #GHz
n_chain = 5
w = np.zeros(mode_num)
gamma_chain = np.zeros(mode_num)
g_chain = np.zeros(mode_num)
g_qubit = 0.1#GHz
chi_chain = np.zeros(mode_num)
kappa_chain = np.zeros(mode_num)
energies = np.zeros(level_num)
C_g_a = 26.0e-18#F, array junction ground capacitance
C_g_b = C_g_a/5.0
C_a = 36.0e-15#F, array junction capacitance, using 45fF/um^2
C_c = 1.0e-15#F, qubit coupling capacitance
C_t = 2.0*(C_g_b + C_c) + (junc_num-1)*C_g_a
E_g_a = 70.0#GHz, array junction ground charging energy
E_j_a = 11.0#GHz, array junction josephson energy
E_j_b = 3 #GHz, qubit junction josephson energy
E_c_a = e**2 / (2*C_a)*1.5e24#GHz, array junction charging energy
E_c_b = 0.835#GHz, qubit junction charging energy
E_t = e**2 / (2*C_t)*1.5e24 #Corresponds to Ct
E_l = 1.0#GHz, qubit inductive energy

#Secondary parameters
lamb = (junc_num-1)*E_t/E_g_a
E_c_phi = ((E_c_b)**-1 + (junc_num*E_c_a)**-1 + (4.0*E_t)**-1*(1.0-2.0/3.0 *(junc_num + 1) / junc_num* lamb))**-1
for idx in range(level_num):
         energies[idx] = bare_hamiltonian(N, E_l, E_c_b, E_j_b, phi_ext*2*np.pi).eigenenergies()[idx]

for i in range(2,mode_num,2):
    E_c_e = (E_c_a ** -1 + (4 * E_g_a * np.sin(np.pi * i / junc_num) ** 2)) ** -1
    g_chain[i] = 4.0/np.sqrt(junc_num) * (E_c_phi*E_c_e/E_g_a) * np.cos(np.pi*i/junc_num)/ np.sin(np.pi*i/junc_num)**2
    w[i] = (8.0*E_c_e*E_j_a)**0.5

    term1 = 0
    term2 = 0
    for level in range(2,level_num):
        term1 = term1 + nem(N,E_l, E_c_b, E_j_b, phi_ext*2*np.pi,0, level)**2* \
                         (energies[level] - energies[0]) / ((energies[level] - energies[0]) ** 2 - w[i] ** 2)
        term2 = term2 + nem(N, E_l, E_c_b, E_j_b, phi_ext * 2 * np.pi, 1, level) ** 2 * \
                        (energies[level] - energies[1]) / ((energies[level] - energies[1]) ** 2 - w[i] ** 2)
    chi_chain[i] = 0.5*np.sqrt(E_j_a/(8.0*E_c_e))*g_chain[i]**2*\
        (nem(N,E_l, E_c_b, E_j_b, phi_ext*2*np.pi,0, 1)**2*2*\
        (energies[1]-energies[0])/((energies[1]-energies[0])**2-w[i]**2)+ term1 + term2)

    l_mode = (8 *E_c_e/E_j_a)**0.25
    kappa_chain[i] = (kappa_r/(w[i]-w_r)**2)*(g_qubit*g_chain[i]/(8*np.sqrt(2)*E_c_phi*l_mode))**2
    # gamma_chain[i] = 4*kappa_chain[i]*chi_chain[i]**2 *n_chain /(kappa_chain[i]**2+4*chi_chain[i]**2)
    gamma_chain[i] = 0.5*kappa_chain[i]*(np.sqrt((1+2j*chi_chain[i]/kappa_chain[i])**2+8j*chi_chain[i]*n_chain/kappa_chain[i])-1).real

plt.figure(1)
plt.semilogy(g_chain, 'd')
plt.tick_params(labelsize = 18.0)
plt.figure(2)
plt.semilogy(chi_chain, 's')
plt.tick_params(labelsize = 18.0)
plt.figure(3)
plt.semilogy(gamma_chain, 'D')
print (np.sum(gamma_chain))
plt.tick_params(labelsize = 18.0)

plt.show()