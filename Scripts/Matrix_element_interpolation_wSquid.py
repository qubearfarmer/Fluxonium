import numpy as np
from matplotlib import pyplot as plt

# plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.figure(figsize=[7, 5])

# Define file directory
directory = 'G:\Projects\Fluxonium\Data\Julius II\Summary'
fname = 'T1 summary 2018_09_28.txt'
path = directory + '\\' + fname
data = np.genfromtxt(path, skip_header=1)  # , delimiter= ',')
current = data[:, 0]
freq = data[:, 1]
T1 = data[:, 2]

directory = "C:\Users\nguyen89\Box\Research\Paper Images\Coherence summary\Fluxonium #33"
fname = "Relaxation_33"
path = directory + "\\" + fname
energies = np.genfromtxt(path + '_energies.txt')
n_element = np.genfromtxt(path + '_chargeElement.txt')
p_element = np.genfromtxt(path + '_fluxElement.txt')
qp_element = np.genfromtxt(path + '_qpElement.txt')
w = energies[:, 1] - energies[:, 0]
matrix_element_interp = np.interp(freq, w, p_element, period=5)

plt.plot(w, matrix_element_interp)
plt.plot(freq, data, '.')
plt.show()
