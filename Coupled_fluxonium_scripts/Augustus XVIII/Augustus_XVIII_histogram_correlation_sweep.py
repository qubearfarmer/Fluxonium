import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*
from scipy.optimize import curve_fit

#constants
kB = 1.38e-23
h = 6.626e-34

f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1205\Histogram_heralding_sweep.hdf5')
d = f.getEntry(0)
for (channel, value) in d.items():
    print(channel, ":", value)

signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
