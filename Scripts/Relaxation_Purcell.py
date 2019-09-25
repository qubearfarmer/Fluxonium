import numpy as np
from qutip import *


# directory = 'C:\\Users\\nguyen89\\Documents'
# fname = '7.5GHz 7mm cavity Purcell LvsFreq.csv'
# path = directory + '\\' + fname
# data = np.genfromtxt(path, skip_header = 1, delimiter = ',')
# inductance = data[1:-1,0]
# mode = data[1:-1,1]
# mode = mode[11:]/1e9
#
# fname = '7.5GHz 7mm cavity Purcell LvsFreq 300nH.csv'
# path = directory + '\\' + fname
# data = np.genfromtxt(path, skip_header = 1, delimiter = ',')
# inductance = np.concatenate((inductance, data[1:-1,0]), axis = 0)
# mode = np.concatenate((mode, data[1:-1,1]/1e9), axis = 0)
#
# fname = '7.5GHz 7mm cavity Purcell LvsQ.csv'
# path = directory + '\\' + fname
# data = np.genfromtxt(path, skip_header = 1, delimiter = ',')
# Q = data[1:-1,1]
# Q = Q[11:]
#
# fname = '7.5GHz 7mm cavity Purcell LvsQ 300nH.csv'
# path = directory + '\\' + fname
# data = np.genfromtxt(path, skip_header = 1, delimiter = ',')
# Q = data[1:-1,1]
# Q = np.concatenate((Q,data[1:-1,1]), axis = 0)
#
# T1_resonator = (mode*2e9*np.pi/Q)**-1.0*1.0e6 #is us
# plt.plot(mode, Q, '.')

# w = np.linspace(4, 7.2, 100)
# Q_interp = np.interp(x=w, xp=mode, fp=Q, period = 4)
# plt.plot(w, Q_interp)

# def charge_matrix_element_resonator(N, w, E_c):
#     E_l = w ** 2 / (8 * E_c)
#     a = tensor(destroy(N))
#     phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
#     na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
#     H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0
#
#     eigen_energies, eigen_states = H.eigenstates()
#     element = na.matrix_element(eigen_states[0], eigen_states[1])
#     return abs(element)


# print (charge_matrix_element_resonator(10, 6.468, 1.017))
# print(6.5e9 / 50000)
# print((6.5e9 / 50000) * (0.228 / 0.63) ** 2)
# nem = np.zeros(len(mode))
# for idx, freq in enumerate(mode):
#     nem[idx] = charge_matrix_element_resonator(30, freq, 0.93)
#
# T1_qubit = T1_resonator*(0.6 / nem)**2
# plt.semilogy(mode, T1_qubit, '.')
# plt.grid()
# plt.show()

g=0.15e9
delta = 1.5e9
kappa = 4e6*2*np.pi
print (((g/delta)**2*kappa)**-1 *1e6)
