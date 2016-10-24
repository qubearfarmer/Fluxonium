from Fluxonium_hamiltonians.Squid_small_junctions import charge_matrix_element as nem
from Fluxonium_hamiltonians.Squid_small_junctions import phase_matrix_element as pem
from Fluxonium_hamiltonians.Squid_small_junctions import qp_matrix_element as qpem
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian as H
import numpy as np
from matplotlib import pyplot as plt

#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Relaxation_wSquid"
path = directory + "\\" + simulation

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34     #Placnk's constant
phi_o = h/(2*e)  #Flux quantum

#######################################################################################
#Qubit and computation parameters
N = 50
E_l = 0.746959655208
E_c = 0.547943694372
E_j_sum = 21.9627179709
level_num = 10
B_coeff = 60
A_j = 3.80888914574e-12
A_c = 1.49982268962e-10
beta_squid = 0.00378012644185
beta_ext = 0.341308382441
d=0.0996032153487
current = np.linspace(0.03,0.05,400)

n_element = np.zeros(len(current))
p_element = np.zeros(len(current))
qp_element = np.zeros((len(current),2))
energies = np.zeros((len(current),level_num))
iState = 0
fState = 1

'''
#######################################################################################
for idx, curr in enumerate(current):
    flux_squid = curr * B_coeff * A_j * 1e-4
    flux_ext = curr * B_coeff * A_c * 1e-4
    n_element[idx]=abs(nem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState))
    p_element[idx] = abs(pem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState))
    qp_element[idx,:] = qpem(N, E_l, E_c, E_j_sum, d, 2 * np.pi * (flux_squid / phi_o - beta_squid),2 * np.pi * (flux_ext / phi_o - beta_ext), iState, fState)
    for idy in range(level_num):
        energies[idx,idy] = H(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),2 * np.pi * (flux_ext / phi_o - beta_ext)).eigenenergies()[idy]

np.savetxt(path+'_energies.txt', energies)
np.savetxt(path+'_chargeElement.txt', n_element)
np.savetxt(path+'_fluxElement.txt', p_element)
np.savetxt(path+'_qpElement.txt', qp_element)
#######################################################################################
'''
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
trans_energy = energies[:,fState]-energies[:,iState]

directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10"
simulation = "T1avg(0to1)vs flux 41p52 to 42p0mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
print data
plt.plot(data[1::,0], data[1::,2], 'ro')
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10"
simulation = "T1_T2_vs YOKO 43p65_45p4mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,0], data[1::,2], 'ro')
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10"
simulation = "T1 avg_T2_qubit f(0to1) vs flux_39p46 to 39p39mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,0], data[1::,2], 'ro')


gamma_cap = np.zeros(len(current))
gamma_ind = np.zeros(len(current))
gamma_qp = np.zeros((len(current),2))
Q_cap = 0.8e6
Q_ind = 0.8e6
Q_qp = 12e6
w = trans_energy*1e9*2*np.pi
hbar = h/(2*np.pi)
kB=1.38064852e-23
T=1e-2
E_c = E_c / 1.509190311677e+24 #convert GHz to J
E_l = E_l / 1.509190311677e+24 #convert to J
E_j_sum = E_j_sum / 1.509190311677e+24 #convert to J
E_j1 = 0.5*E_j_sum*(1 + d)
E_j2 = 0.5*E_j_sum*(1 - d)
delta_alum = 5.447400321e-23 #J

cap = e**2/(2.0*E_c)
ind = hbar**2/(4.0*e**2*E_l)
gk = e**2.0/h
g1 = 8.0*E_j1*gk/delta_alum
g2 = 8.0*E_j2*gk/delta_alum
Y_cap = w*cap/Q_cap
Y_ind = 1.0/(w*ind*Q_ind)
Y_qp1 = (g1/(2*Q_qp))*(2*delta_alum/(hbar*w))**(1.5)
Y_qp2 = (g2/(2*Q_qp))*(2*delta_alum/(hbar*w))**(1.5)

R_cap = 1.0/(w*cap*Q_cap)
R_ind = (w*ind/Q_ind)
for idx in range(len(current)):
    gamma_cap[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_cap[idx] * (1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    gamma_ind[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_ind[idx] * (1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    gamma_qp[idx,0] = (qp_element[idx,0]) ** 2 * (w[idx] / np.pi / gk) * Y_qp1[idx]
    gamma_qp[idx, 1] = (qp_element[idx, 1]) ** 2 * (w[idx] / np.pi / gk) * Y_qp2[idx]
plt.semilogy(current*1e3 + (41.6813-41.6413),1/gamma_cap*1e6, 'b-', current*1e3+ (41.6813-41.6413), 1/gamma_qp[:,0]*1e6, 'r-',current*1e3+ (41.6813-41.6413), 1/1/gamma_qp[:,1]*1e6, 'g-',current*1e3+ (41.6813-41.6413), 1/1/gamma_ind*1e6, 'y-')
plt.grid()
plt.xlabel("Current (mA")
plt.ylabel("T1(us)")
plt.show()
