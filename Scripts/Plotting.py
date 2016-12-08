#This script is used to plot simulated data, with the purpose of keeping the original scripts intact
import numpy as np
from matplotlib import pyplot as plt

#Define file directory
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "MElements_wSquid"
path = directory + "\\" + simulation
current = np.linspace(0.035, 0.046, 1001)
iState = 0
fState = 1
path = path+'_'+str(iState)+'to'+str(fState)+'_from_' + str(current[0]*1e3) +'to'+ str(current[-1]*1e3) +'mA'
energies = np.genfromtxt(path+'_energies.txt')
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element1 = np.genfromtxt(path + '_phaseElement.txt')
qp_element1 = np.genfromtxt(path + '_qpElement.txt')
# plt.plot(current*1e3,p_element1**2, 'b-')
# plt.plot(current*1e3, qp_element1[:,0]**2+qp_element1[:,1]**2, 'b-')

iState = 0
fState = 2
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "MElements_wSquid"
path = directory + "\\" + simulation
path = path+'_'+str(iState)+'to'+str(fState)+'_from_' + str(current[0]*1e3) +'to'+ str(current[-1]*1e3) +'mA'
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element = np.genfromtxt(path+'_phaseElement.txt')
qp_element = np.genfromtxt(path+'_qpElement.txt')

# plt.plot(current*1e3, qp_element[:,0]**2+qp_element[:,1]**2, 'b--')

iState = 1
fState = 2
directory = "C:\Data\Fluxonium #10 simulations"
simulation = "MElements_wSquid"
path = directory + "\\" + simulation
path = path+'_'+str(iState)+'to'+str(fState)+'_from_' + str(current[0]*1e3) +'to'+ str(current[-1]*1e3) +'mA'
n_element = np.genfromtxt(path+'_chargeElement.txt')
p_element2 = np.genfromtxt(path + '_phaseElement.txt')
qp_element2 = np.genfromtxt(path + '_qpElement.txt')
plt.plot(current * 1e3, p_element1 ** 2 / (qp_element1[:, 0] ** 2 + qp_element1[:, 1] ** 2), 'b-')
plt.plot(current * 1e3, p_element2 ** 2 / (qp_element2[:, 0] ** 2 + qp_element2[:, 1] ** 2), 'r-')

plt.show()
