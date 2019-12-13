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

def osc_func(x,amp,freq,offset1,offset2):
    return amp * np.cos(2 * np.pi * freq * (x - offset1)) - offset2

f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1212\Ramsey_heralded_qubit_A.hdf5')

