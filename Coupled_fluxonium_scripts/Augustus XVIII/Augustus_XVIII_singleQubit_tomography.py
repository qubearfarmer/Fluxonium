import numpy as np
from qutip import *
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

##############single qubit tomography##############
#beta calibration

#Gate sequence in Labber is I, X2p, Y2m
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1214\Tomography_qubit_A_I.hdf5')
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
preselected_data = np.zeros(len(signal[:,0]), dtype = complex)
xmin = -350
xmax = -100
ymin = -1000
ymax = -750

for pulse_idx in range(len(signal[:,0])):
    preselected_signal = []
    herald_signal = signal[pulse_idx, 0::2]*1e6
    select_signal = signal[pulse_idx, 1::2]*1e6
    for record_idx in range(len(herald_signal)):
        if (xmin <= np.real(herald_signal[record_idx]) <= xmax) and (ymin <= np.imag(herald_signal[record_idx]) <= ymax):
            preselected_signal = np.append(preselected_signal, select_signal[record_idx])
    preselected_data[pulse_idx] = np.average(preselected_signal)

s0 = preselected_data[0]
sz = preselected_data[6]
# print (s0,sz)
betaI = s0 + sz
betaZ = s0 - sz
#Start in ground state
m = preselected_data[0:3]

measurement_matrix = 0.5*np.array([[0, 0, betaZ], [0, betaZ, 0], [betaZ, 0, 0]])
avgX, avgY, avgZ = np.linalg.inv(measurement_matrix).dot(m.transpose()-0.5*betaI).transpose()
rho_reconstructed = 0.5*(qeye(2) + avgX*sigmax() + avgY*sigmay() + avgZ*sigmaz())
matrix_histogram_complex(rho_reconstructed)
rho_ideal = ket2dm(basis(2,0))
plt.title(fidelity(rho_ideal, rho_reconstructed))
# matrix_histogram_complex(rho_ideal)
#Start in excited state
m = preselected_data[6:9]
avgX, avgY, avgZ = np.linalg.inv(measurement_matrix).dot(m.transpose()-0.5*betaI).transpose()
rho_reconstructed = 0.5*(qeye(2) + avgX*sigmax() + avgY*sigmay() + avgZ*sigmaz())
matrix_histogram_complex(rho_reconstructed)
rho_ideal = ket2dm(basis(2,1))
plt.title(fidelity(rho_ideal, rho_reconstructed))
# matrix_histogram_complex(rho_ideal)
#Start in superposition state
m = preselected_data[9:]
avgX, avgY, avgZ = np.linalg.inv(measurement_matrix).dot(m.transpose()-0.5*betaI).transpose()
rho_reconstructed = 0.5*(qeye(2) + avgX*sigmax() + avgY*sigmay() + avgZ*sigmaz())
matrix_histogram_complex(rho_reconstructed)
rho_ideal = ket2dm(rx(phi=np.pi/2)*basis(2,0))
plt.title(fidelity(rho_ideal, rho_reconstructed))
# matrix_histogram_complex(rho_ideal)

#####################################################################################################
# sI = np.array([[1,0],[0,1]])
# sX = np.array([[0,1],[1,0]])
# sY = np.array([[0,-1j],[1j,0]])
# sZ = np.array([[1,0],[0,-1]])
#
# M = np.zeros((3, 2, 2), dtype = complex)
# M[0,:,:] = betaI +betaZ*sZ
# M[1,:,:] = betaI +betaZ*sY
# M[2,:,:] = betaI +betaZ*sX

# def density_matrix(t1,t2,t3):
#     tau = np.array([[t1, t2+1j*t3], [0, 1-t1]])
#     rho = np.conj(tau.transpose())*tau / np.trace(np.conj(tau.transpose())*tau)
#     return rho
#
# def likelihood(x):
#     dist = 0
#     for idx in range(3):
#         dist = dist + np.real((m[idx] - np.trace(M[idx, :, :].dot(density_matrix(x[0],x[1],x[2])))))**2 + np.imag((m[idx] - np.trace(M[idx, :, :].dot(density_matrix(x[0],x[1],x[2])))))**2
#     return dist
#
# guess = np.ones(3)*0.5
# # guess[0] = 1
# res = minimize(likelihood, guess, method='nelder-mead')
# t = res.x
# rho_reconstructed_mle = density_matrix(*t)
# matrix_histogram_complex(rho_reconstructed_mle)
plt.show()