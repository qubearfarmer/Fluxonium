import numpy as np
from matplotlib import pyplot as plt

# plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.figure(figsize=[7, 5])

# Define file directory
directory = 'C:\\Users\\nguyen89\\Box\Research\\Paper Images\\Coherence summary\\Fluxonium #33'
fname = 'T1 summary 2018_09_28.txt'
path = directory + '\\' + fname
data = np.genfromtxt(path, skip_header=1)  # , delimiter= ',')
current = data[:, 0]
freq = data[:, 1]
T1 = data[:, 2]

directory = "C:\\Users\\nguyen89\Box\Python Codes\\Fluxonium simulation results"
fname = "Relaxation_33"
path = directory + "\\" + fname
energies = np.genfromtxt(path + '_energies.txt')
n_element = np.genfromtxt(path + '_chargeElement.txt')
p_element = np.genfromtxt(path + '_fluxElement.txt')
qp_element = np.genfromtxt(path + '_qpElement.txt')
w = energies[:, 1] - energies[:, 0]
matrix_element_interp = np.interp(freq, w, p_element, period=6)

# directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Summary'
# fname = 'T1_60to62mA.txt'
# path = directory + '\\'+ fname
# data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
# current = data[:,0]
# freq = data[:,1]
# T1 = data[:,2]
#
# directory = "C:\\Users\\nguyen89\Documents\\Fluxonium simulation"
# fname = "Relaxation_13"
# path = directory + "\\" + fname
# energies = np.genfromtxt(path+'_energies.txt')
# n_element = np.genfromtxt(path+'_chargeElement.txt')
# p_element = np.genfromtxt(path+'_fluxElement.txt')
# qp_element = np.genfromtxt(path+'_qpElement.txt')
# w = energies[:,1] - energies[:,0]
# matrix_element_interp = np.interp(freq, w, p_element, period = 3.8)

# directory = 'G:\Projects\Fluxonium\Data\Fluxonium #28\Summary'
# fname = 'T1_summary_2018_04_13.txt'
# path = directory + '\\'+ fname
# data = np.genfromtxt(path, skip_header = 1)#, delimiter= ',')
# current = data[:,0]
# freq = data[:,1]
# T1 = data[:,2]
#
# directory = "C:\\Users\\nguyen89\Documents\\Fluxonium simulation"
# fname = "Relaxation_28"
# path = directory + "\\" + fname
# energies = np.genfromtxt(path+'_energies.txt')
# n_element = np.genfromtxt(path+'_chargeElement.txt')
# p_element = np.genfromtxt(path+'_fluxElement.txt')
# qp_element = np.genfromtxt(path+'_qpElement.txt')
# w = energies[:,1] - energies[:,0]
# matrix_element_interp = np.interp(freq, w, p_element, period = 6)

# Check that the interpolation works
plt.plot(freq, matrix_element_interp, '.', w, p_element)
np.savetxt(path + '_matrix_element_interp.txt', matrix_element_interp)
plt.show()
