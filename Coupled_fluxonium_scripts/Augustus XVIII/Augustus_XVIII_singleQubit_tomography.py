import numpy as np
from qutip import *
from matplotlib import pyplot as plt
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

##############single qubit tomography##############
#beta calibration
# sz = (-495+1j*90) #qubit B
# s0 = (-220-1j*816)
# sz = (-506+1j*230) #qubit B ideal
# s0 = (-230-1j*686)

sz = (56-1j*445)  #qubit A
s0 = (-225-1j*715)
# sz = (129-1j*444)  #qubit A ideal
# s0 = (-220-1j*816)
betaI = s0  + sz
betaZ = s0 - sz
#Gate sequence in Labber is I, X2p, Y2m
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1211\Tomography_qubit_A_exc.hdf5')
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
m = np.zeros(3, dtype = complex)
xmin = -250
xmax = 0
ymin = -1000
ymax = -700

#Expected state
# rho_ideal = ket2dm(basis(2,0))
rho_ideal = ket2dm(ry(phi=np.pi/2)*basis(2,0))
# rho_ideal = ket2dm(basis(2,1))

for pulse_idx in range(3):
    preselected_signal = []
    herald_signal = signal[pulse_idx, 0::2]*1e6
    select_signal = signal[pulse_idx, 1::2]*1e6
    for record_idx in range(len(herald_signal)):
        if (xmin <= np.real(herald_signal[record_idx]) <= xmax) and (ymin <= np.imag(herald_signal[record_idx]) <= ymax):
            preselected_signal = np.append(preselected_signal, select_signal[record_idx])
    m[pulse_idx] = np.average(preselected_signal)

measurement_matrix = 0.5*np.array([[0, 0, betaZ], [0, betaZ, 0], [betaZ, 0, 0]])
# # print (measurement_matrix)
avgX, avgY, avgZ = np.linalg.inv(measurement_matrix).dot(m.transpose()-0.5*betaI).transpose()
rho_reconstructed = 0.5*(qeye(2) + avgX*sigmax() + avgY*sigmay() + avgZ*sigmaz())
matrix_histogram_complex(rho_reconstructed)
print (fidelity(rho_ideal, rho_reconstructed))
matrix_histogram_complex(rho_ideal)
plt.show()