from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
import numpy as np
from matplotlib import pyplot as plt

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

#######################################################################################
N = 50
E_l = 0.46
E_c = 3.6
E_j = 10.2
level_num = 10
iState = 0
fState = 1

phi_ext = np.linspace(-0.05,0.55,601)
p_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))
# '''
#######################################################################################
for idx, phi in enumerate(phi_ext):
    p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2*np.pi, iState, fState))
    qp_element[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2 * np.pi, iState, fState))
    for idy in range(level_num):
        energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[idy]

trans_energy = energies[:,fState]-energies[:,iState]
#######################################################################################
# '''
print ("Step 1")
gamma_cap = np.zeros(len(phi_ext))
gamma_ind = np.zeros(len(phi_ext))
Q_cap = 3.0e6
Q_ind = 10e6
Q_qp = 0.3e6
w = trans_energy*1e9*2*np.pi
hbar = h/(2*np.pi)
kB=1.38064852e-23
T=1e-2
E_c = E_c / 1.509190311677e+24 #convert GHz to J
E_l = E_l / 1.509190311677e+24 #convert to J
E_j = E_j / 1.509190311677e+24 #convert to J
delta_alum = 5.447400321e-23 #J

cap = e**2/(2*E_c)
ind = hbar**2/(4*e**2*E_l)
Gt = 8*E_j*e**2/(delta_alum*h)
Q_cap = 3.0e6
Y_cap = w*cap/Q_cap
Y_ind = 1.0/(w*ind*Q_ind)
Y_qp = (Gt/(2*Q_qp))*(2*delta_alum/(hbar*w))**(1.5)
print ("Step 2")
for idx in range(len(phi_ext)):
    gamma_cap[idx] = (phi_o*p_element[idx]/hbar/(2*np.pi))**2*hbar*w[idx]*Y_cap[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
    gamma_ind[idx] = (phi_o*p_element[idx]/hbar/(2*np.pi))**2*hbar*w[idx]*Y_ind[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
phi_ext_alt = phi_ext[0:-5]
gamma_qp = np.zeros(len(phi_ext_alt))
# for idx in range(len(phi_ext_alt)):
#     gamma_qp[idx] = (qp_element[idx])**2*w[idx]*Y_qp[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
plt.semilogy(phi_ext,1/gamma_cap*1e6)

Q_cap = 1.0e6
Y_cap = w*cap/Q_cap
Y_ind = 1.0/(w*ind*Q_ind)
Y_qp = (Gt/(2*Q_qp))*(2*delta_alum/(hbar*w))**(1.5)
print ("Step 2")
for idx in range(len(phi_ext)):
    gamma_cap[idx] = (phi_o*p_element[idx]/hbar/(2*np.pi))**2*hbar*w[idx]*Y_cap[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
    gamma_ind[idx] = (phi_o*p_element[idx]/hbar/(2*np.pi))**2*hbar*w[idx]*Y_ind[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
phi_ext_alt = phi_ext[0:-5]
gamma_qp = np.zeros(len(phi_ext_alt))
# for idx in range(len(phi_ext_alt)):
#     gamma_qp[idx] = (qp_element[idx])**2*w[idx]*Y_qp[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
plt.semilogy(phi_ext,1/gamma_cap*1e6)

Q_cap = 10.0e6
Y_cap = w*cap/Q_cap
Y_ind = 1.0/(w*ind*Q_ind)
Y_qp = (Gt/(2*Q_qp))*(2*delta_alum/(hbar*w))**(1.5)
print ("Step 2")
for idx in range(len(phi_ext)):
    gamma_cap[idx] = (phi_o*p_element[idx]/hbar/(2*np.pi))**2*hbar*w[idx]*Y_cap[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
    gamma_ind[idx] = (phi_o*p_element[idx]/hbar/(2*np.pi))**2*hbar*w[idx]*Y_ind[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
phi_ext_alt = phi_ext[0:-5]
gamma_qp = np.zeros(len(phi_ext_alt))
# for idx in range(len(phi_ext_alt)):
#     gamma_qp[idx] = (qp_element[idx])**2*w[idx]*Y_qp[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
plt.semilogy(phi_ext,1/gamma_cap*1e6)


plt.ylim([0.1,1.1e5])
# plt.semilogy(phi_ext_alt,1/gamma_qp*1e6)
plt.grid()
plt.show()
