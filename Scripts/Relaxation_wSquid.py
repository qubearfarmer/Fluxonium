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
current = np.linspace(0.038, 0.046, 801)


n_element = np.zeros(len(current))
p_element = np.zeros(len(current))
qp_element = np.zeros((len(current),2))
energies = np.zeros((len(current),level_num))

iState = 0
fState = 1
path = path+'_'+str(iState)+'to'+str(fState)+'_from_' + str(current[0]*1e3) +'to'+ str(current[-1]*1e3) +'mA'
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
print ("Simulated data saved")
#######################################################################################
# '''
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')

# '''
#T1 data for 01 transition
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Corrected flux"

simulation = "T1avg(0to1)vs flux 41p52 to 42p0mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,1], data[1::,2], 'ro')

simulation = "T1_T2_qubit f(0to1)vs flux 43p65_45p4mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,1], data[1::,2], 'ro')

simulation = "T1 avg_T2_qubit f(0to1) vs flux_39p46 to 39p39mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,1], data[1::,2], 'ro')

simulation = "T1 avg_T2_qubit f(0to1) vs flux_38p5 to 38p76mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,1], data[1::,2], 'ro')
'''
#Automation T1
#T1 data for 01 transition
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Automation code"

simulation = "T1_auto_35to36p1mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
plt.plot(data[:,0], data[:,2], 'r.')

simulation = "T1_auto_37p2to38p5mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
plt.plot(data[:,0], data[:,2], 'r.')

simulation = "T1_auto_39p5to40p7mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
plt.plot(data[:,0], data[:,2], 'r.')

simulation = "T1_auto_41p8to42p9mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
plt.plot(data[:,0], data[:,2], 'r.')

simulation = "T1_auto_44to45mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path)
plt.plot(data[:,0], data[:,2], 'r.')

#T1 data for 02 transition
directory = "G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Corrected flux"

simulation = "T1 avg_T2_qubit f(0to2) vs flux_38p76 to 38p26mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,0], data[1::,2], 'ro')

simulation = "T1 avg_T2_qubit f(0to2) vs flux_39p37 to 39p78mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,0], data[1::,2], 'ro')

simulation = "T1 avg_T2_qubit f(0to2) vs flux_41p5 to 42mA.csv"
path = directory + "\\" + simulation
data = np.genfromtxt(path, delimiter =',',dtype=float)
plt.plot(data[1::,0], data[1::,2], 'ro')
'''

hbar = h/(2*np.pi)
kB=1.38064852e-23
T=1e-2
E_c = E_c / 1.509190311677e+24 #convert GHz to J
E_l = E_l / 1.509190311677e+24 #convert to J
E_j_sum = E_j_sum / 1.509190311677e+24 #convert to J
E_j1 = 0.5*E_j_sum*(1 + d)
E_j2 = 0.5*E_j_sum*(1 - d)
delta_alum = 5.447400321e-23 #J

Q_cap = 3e6
Q_ind = 0.8e6
Q_qp1 = 20e6
Q_qp2 = 20e6

cap = e**2/(2.0*E_c)
ind = hbar**2/(4.0*e**2*E_l)
gk = e**2.0/h
g1 = 8.0*E_j1*gk/delta_alum
g2 = 8.0*E_j2*gk/delta_alum

trans_energy = energies[:,fState]-energies[:,iState]
w = trans_energy*1e9*2*np.pi

Y_cap = w*cap/Q_cap
Y_ind = 1.0/(w*ind*Q_ind)
Y_qp1 = (g1/(2*Q_qp1))*(2*delta_alum/(hbar*w))**(1.5)
Y_qp2 = (g2/(2*Q_qp2))*(2*delta_alum/(hbar*w))**(1.5)

gamma_cap1 = np.zeros(len(current))
gamma_ind1 = np.zeros(len(current))
gamma_qp1 = np.zeros((len(current),2))

for idx in range(len(current)):
    gamma_cap1[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_cap[idx] * (1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    gamma_ind1[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_ind[idx] * (1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    gamma_qp1[idx,0] = (qp_element[idx,0]) ** 2 * (w[idx] / np.pi / gk) * Y_qp1[idx]
    gamma_qp1[idx, 1] = (qp_element[idx, 1]) ** 2 * (w[idx] / np.pi / gk) * Y_qp2[idx]
gamma = gamma_qp1[:,0] + gamma_qp1[:,1]
# plt.semilogy(current*1e3 ,1.0/gamma_cap1*1e6,'b-')
# plt.semilogy(current*1e3 ,1.0/gamma_qp1[:,0]*1e6,'g-')
# plt.semilogy(current*1e3 ,1.0/gamma_qp1[:,1]*1e6,'m-')
# plt.semilogy(current*1e3 ,1.0/gamma*1e6,'g-')
# plt.semilogy(current*1e3 ,1.0/gamma_ind1*1e6,'y-')
# plt.ylim([1,10e3])
'''
######################################################################################
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Relaxation_wSquid"
path = directory + "\\" + simulation
iState = 1
fState = 2
path = path+"_" + str(iState)+ str(fState)
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_fluxElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')
trans_energy = energies[:,fState]-energies[:,iState]
w = trans_energy*1e9*2*np.pi

Q_cap = 0.45e6
Q_ind = 50e5
Q_qp1 = 50e6
Q_qp2 = 70e6

Y_cap = w*cap/Q_cap
Y_ind = 1.0/(w*ind*Q_ind)
Y_qp1 = (g1/(2*Q_qp1))*(2*delta_alum/(hbar*w))**(1.5)
Y_qp2 = (g2/(2*Q_qp2))*(2*delta_alum/(hbar*w))**(1.5)

gamma_cap2 = np.zeros(len(current))
gamma_ind2 = np.zeros(len(current))
gamma_qp2 = np.zeros((len(current),2))

for idx in range(len(current)):
    gamma_cap2[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_cap[idx] * (1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    gamma_ind2[idx] = (phi_o * p_element[idx] / hbar / (2 * np.pi)) ** 2 * hbar * w[idx] * Y_ind[idx] * (1 + 1.0 / np.tanh(hbar * w[idx] / (2 * kB * T)))
    gamma_qp2[idx,0] = (qp_element[idx,0]) ** 2 * (w[idx] / np.pi / gk) * Y_qp1[idx]
    gamma_qp2[idx, 1] = (qp_element[idx, 1]) ** 2 * (w[idx] / np.pi / gk) * Y_qp2[idx]

gamma_cap = gamma_cap1 + gamma_cap2
gamma_ind = gamma_ind1 + gamma_ind2
gamma_qp = gamma_qp1 + gamma_qp2
gamma = gamma_cap + gamma_ind + gamma_qp1[:,0] + gamma_qp1[:,1] + gamma_qp2[:,0] + gamma_qp2[:,1]
'''
# plt.semilogy(current*1e3 + (41.6813-41.6413),1.0/gamma*1e6,'k-')
plt.grid()
# plt.xlim([current[0]*1e3,current[-1]*1e3])
# plt.xlabel("Current (mA")
# plt.ylabel("T1(us)")
plt.show()
