import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1205\Histogram_3.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)

def gaussian4(x,a1,x1,a2,x2,a3,x3,a4,x4,sigma):
    return a1*np.exp(-(x-x1)**2/sigma**2) + a2*np.exp(-(x-x2)**2/sigma**2) + \
           a3*np.exp(-(x-x3)**2/sigma**2) + a4*np.exp(-(x-x4)**2/sigma**2)

signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
# delay_time = f.getData('Multi-Qubit Pulse Generator - Delay after heralding')[:,0]
repetition = 10
signal = signal[0]
sReal = np.real(signal)*1e6
sImag = np.imag(signal)*1e6
H, xedges, yedges = np.histogram2d(sReal,sImag, bins = [100,100])
H = H.T
##########################################################
plt.figure(1)
X,Y = np.meshgrid(xedges, yedges)
plt.pcolormesh(X,Y,H)

##########################################################
plt.figure(2)
counts, edges = np.histogram(sReal, bins = 100)
plt.plot(edges[:-1], counts)
plt.show()