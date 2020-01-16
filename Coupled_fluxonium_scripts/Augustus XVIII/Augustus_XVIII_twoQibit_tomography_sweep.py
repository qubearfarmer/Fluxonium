import numpy as np
from qutip import *
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

##############single qubit tomography##############
#beta calibration
f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2020\\01\Data_0115\Tomography_twoQubits_ROCal.hdf5')
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
s = np.zeros(4, dtype = complex)
#old data
# xmin = -350
# xmax = -100
# ymin = -1000
# ymax = -700

# xmin = -30
# xmax = 250
# ymin = -700
# ymax = -300

#00
xmin = 172
xmax = 340
ymin = -20
ymax = 160

for pulse_idx in range(4):
    preselected_signal = []
    herald_signal = signal[pulse_idx, 0::2]*1e6
    select_signal = signal[pulse_idx, 1::2]*1e6
    sReal = np.real(herald_signal)
    sImag = np.imag(herald_signal)
    # if pulse_idx == 0:
    #     H, xedges, yedges = np.histogram2d(sReal, sImag, bins=[100, 100])
    #     H = H.T
    #     X, Y = np.meshgrid(xedges, yedges)
    #     plt.pcolormesh(X, Y, H, cmap='GnBu')
    for record_idx in range(len(herald_signal)):
        if (xmin <= np.real(herald_signal[record_idx]) <= xmax) and (ymin <= np.imag(herald_signal[record_idx]) <= ymax):
            preselected_signal = np.append(preselected_signal, select_signal[record_idx])
    s[pulse_idx] = np.average(preselected_signal)
sII, sZI, sIZ, sZZ = s
sMatrix = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
betaII, betaZI, betaIZ, betaZZ = np.linalg.inv(sMatrix).dot(np.array([sII,sZI,sIZ,sZZ]).transpose()).transpose()

##############################################################
def density_matrix(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15):
    t16 = 1 - t1 -t2 - t3
    tau = np.array([[t1, t4+1j*t5, t6+1j*t7, t8+1j*t9], [0, t2, t10+1j*t11, t12+1j*t13], [0, 0, t3, t14+1j*t15], [0,0,0,t16]])
    rho = np.conj(tau.transpose()).dot(tau)
    rho = rho / np.trace(rho)
    return rho

def likelihood(x):
    dist = 0
    for idx in range(len(gate_sequence)):
        dist = dist + abs((m[idx] - np.trace(M[idx, :, :].dot(density_matrix(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10], x[11],x[12],x[13],x[14])))))**2
    return dist
def fidelity(rho,rho_ideal):
    return abs(np.trace(np.sqrt(np.sqrt(rho_ideal).dot(rho).dot(np.sqrt(rho_ideal)))))

#Maximum likelihood
gate_sequence = np.array(['I-I','Xp-I','I-Xp',\
                          'X2p-I','X2p-X2p','X2p-Y2p','X2p-Xp',\
                          'Y2p-I','Y2p-X2p','Y2p-Y2p','Y2p-Xp',\
                          'I-X2p','Xp-X2p','I-Y2p','Xp-Y2p'])
sI = np.array([[1,0],[0,1]])
sX = np.array([[0,1],[1,0]])
sY = np.array([[0,-1j],[1j,0]])
sZ = np.array([[1,0],[0,-1]])

II = np.kron(sI,sI)
IX = np.kron(sI,sX)
IY = np.kron(sI,sY)
IZ = np.kron(sI,sZ)
XI = np.kron(sX,sI)
XX = np.kron(sX,sX)
XY = np.kron(sX,sY)
XZ = np.kron(sX,sZ)
YI = np.kron(sY,sI)
YX = np.kron(sY,sX)
YY = np.kron(sY,sY)
YZ = np.kron(sY,sZ)
ZI = np.kron(sZ,sI)
ZX = np.kron(sZ,sX)
ZY = np.kron(sZ,sY)
ZZ = np.kron(sZ,sZ)

M = np.zeros((len(gate_sequence), 4, 4), dtype = complex)
M[0,:,:] = betaII*II + betaZI*ZI + betaIZ*IZ + betaZZ*ZZ
M[1,:,:] = betaII*II - betaZI*ZI + betaIZ*IZ - betaZZ*ZZ
M[2,:,:] = betaII*II + betaZI*ZI - betaIZ*IZ - betaZZ*ZZ

M[3,:,:] = betaII*II + betaZI*YI + betaIZ*IZ + betaZZ*YZ
M[4,:,:] = betaII*II + betaZI*YI + betaIZ*IY + betaZZ*YY
M[5,:,:] = betaII*II + betaZI*YI - betaIZ*IX - betaZZ*YX
M[6,:,:] = betaII*II + betaZI*YI - betaIZ*IZ - betaZZ*YZ

M[7,:,:] = betaII*II - betaZI*XI + betaIZ*IZ - betaZZ*XZ
M[8,:,:] = betaII*II - betaZI*XI + betaIZ*IY - betaZZ*XY
M[9,:,:] = betaII*II - betaZI*XI - betaIZ*IX + betaZZ*XX
M[10,:,:] = betaII*II - betaZI*XI - betaIZ*IZ + betaZZ*XZ

M[11,:,:] = betaII*II + betaZI*ZI + betaIZ*IY + betaZZ*ZY
M[12,:,:] = betaII*II - betaZI*ZI + betaIZ*IY - betaZZ*ZY
M[13,:,:] = betaII*II + betaZI*ZI - betaIZ*IX - betaZZ*ZX
M[14,:,:] = betaII*II - betaZI*ZI - betaIZ*IX + betaZZ*ZX

################################################################
#Tomography data
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2020\\01\Data_0115\Tomography_twoQubit_sup_phasesweep.hdf5')
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
delay = f.getData('Multi-Qubit Pulse Generator - Plateau #3')[:,0]
m = np.zeros(15, dtype = complex)
guess_mle = np.zeros(15)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
print (np.round(delay*1e9,3))
#11
xmin = -174
xmax = -16
ymin = -376
ymax = -204
for delay_idx in range(len(delay)):
    for pulse_idx in range(15):
        preselected_signal = []
        herald_signal = signal[delay_idx*15+pulse_idx, 0::2]*1e6
        select_signal = signal[delay_idx*15+pulse_idx, 1::2]*1e6
        for record_idx in range(len(herald_signal)):
            if (xmin <= np.real(herald_signal[record_idx]) <= xmax) and (ymin <= np.imag(herald_signal[record_idx]) <= ymax):
                preselected_signal = np.append(preselected_signal, select_signal[record_idx])
        m[pulse_idx] = np.average(preselected_signal)

    res = minimize(likelihood, guess_mle, method='powell',tol=1.e-10,
                options={'maxiter': 10000})
    t = res.x
    rho_reconstructed_mle = Qobj(density_matrix(*t))
    matrix_histogram_complex(rho_reconstructed_mle)
    plt.title(str(np.round(delay[delay_idx]*1e9, 3)))
    fname = 'C:\Data\Projects\Fluxonium\Data\Augustus 18\\tomography_sweep_3_' + str(delay_idx)
    plt.savefig(fname)