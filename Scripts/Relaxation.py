from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian as H
import numpy as np
from matplotlib import pyplot as plt

#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Relaxation"
path = directory + "\\" + simulation

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
n_element = np.zeros(len(phi_ext))
qp_element = np.zeros(len(phi_ext))
energies = np.zeros((len(phi_ext),level_num))
'''
#######################################################################################
for idx, phi in enumerate(phi_ext):
    p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2*np.pi, iState, fState))
    n_element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2*np.pi, iState, fState))
    qp_element[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2 * np.pi, iState, fState))
    for idy in range(level_num):
        energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2*np.pi).eigenenergies()[idy]

np.savetxt(path + '_energies.txt', energies)
np.savetxt(path + '_chargeElement.txt', n_element)
np.savetxt(path + '_fluxElement.txt', p_element)
np.savetxt(path + '_qpElement.txt', qp_element)
######################################################################################
'''
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')

trans_energy = energies[:,fState]-energies[:,iState]

gamma_cap = np.zeros(len(phi_ext))
gamma_ind = np.zeros(len(phi_ext))
gamma_qp = np.zeros(len(phi_ext))
Q_cap = 1
Q_ind = 1
Q_qp = 1
w = trans_energy*1e9*2*np.pi
hbar = h/(2*np.pi)
kB=1.38064852e-23
T=1e-2
E_c = E_c / 1.509190311677e+24 #convert GHz to J
E_l = E_l / 1.509190311677e+24 #convert to J
E_j = E_j / 1.509190311677e+24 #convert to J
delta_alum = 5.447400321e-23 #J

cap = e**2/(2.0*E_c)
ind = hbar**2/(4.0*e**2*E_l)
gk = e**2.0/h
g = 8.0*E_j*gk/delta_alum
Y_cap = w*cap/Q_cap
Y_ind = 1.0/(w*ind*Q_ind)
Y_qp = (g/(2*Q_qp))*(2*delta_alum/(hbar*w))**(1.5)
print ("Step 2")
for idx in range(len(phi_ext)):
    gamma_cap[idx] = (phi_o*p_element[idx]/hbar/(2*np.pi))**2*hbar*w[idx]*Y_cap[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
    gamma_ind[idx] = (phi_o*p_element[idx]/hbar/(2*np.pi))**2*hbar*w[idx]*Y_ind[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
    gamma_qp[idx] = (qp_element[idx]) ** 2 *(w[idx]/np.pi/gk)*Y_qp[idx]

plt.semilogy(phi_ext,1/gamma_cap*1e6,phi_ext,1/gamma_ind*1e6)
for idx in range(len(phi_ext)):
    if gamma_qp[idx] == 0:
        gamma_qp_alt=np.delete(gamma_qp, idx)
        phi_ext_alt = np.delete(phi_ext, idx)
plt.semilogy(phi_ext_alt,1/gamma_qp_alt*1e6)

###########################################################
phi_ext = np.linspace(-0.05,0.55,601)
R_cap = 1.0/(w*cap*Q_cap)
R_ind = (w*ind/Q_ind)
for idx in range(len(phi_ext)):
    gamma_cap[idx] = (2*e*n_element[idx]/hbar)**2*hbar*w[idx]*R_cap[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
    gamma_ind[idx] = (2*e*n_element[idx]/hbar)**2*hbar*w[idx]*R_ind[idx]*(1+1.0/np.tanh(hbar*w[idx]/(2*kB*T)))
# plt.semilogy(phi_ext,1/gamma_cap*1e6, 'b--', phi_ext,1/gamma_ind*1e6, 'g--')
ratio = (n_element / p_element)**2
rev = 1/w**2
# plt.semilogy(phi_ext, ratio, phi_ext, w**2 * ratio[0]/w[0]**2, phi_ext, rev*ratio[0]/rev[0])
plt.grid()
plt.show()
