import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

f = Labber.LogFile('G:\Projects\Fluxonium\Data\AugustusXVI\\2019\\09\Data_0927\\083bis2.hdf5')
d = f.getEntry(0)
for (channel, value) in d.items():
    print(channel, ":", value)
time = f.getData('Multi-Qubit Pulse Generator - Width #1')[0]
freq = f.getData('Qubit RF - Frequency')[:,0]
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
X,Y = np.meshgrid(time*1e6,freq*1e-6)
Z = signal_real
plt.pcolormesh(X,Y,Z, cmap = 'GnBu')
plt.tick_params(labelsize = 18.0)
plt.xlim([0,0.5])
plt.show()



