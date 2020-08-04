import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*
from scipy.optimize import curve_fit

def randomized_benchmarking_0(x,p,a,b):
    return a*p**x+b

def randomized_benchmarking_1(x,p,a,b,c):
    return a*p**x+b +c*(x-1)*p**(x-2)


cliffNum = np.array([  2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  20.,  24.,  28.,  32.,  40.,  48., 56.,  64.,  80.,  96., 112., 128., 160., 192., 224., 256.])
nicex = np.linspace(cliffNum[0], cliffNum[-1], 101)
path = 'C:\\Users\\nguyen89\Documents\Python Codes\Data analysis\RB data\Augustus18_QubitA_RB.txt'
data = np.genfromtxt(path)
plt.errorbar(cliffNum,data, fmt='o', mfc='none', mew=2.0)
guess =([0.99,np.max(data)-np.min(data),np.min(data), 1])
opt,cov = curve_fit(randomized_benchmarking_1, ydata = data, xdata = cliffNum, p0 = guess)
plt.plot(nicex, randomized_benchmarking_1(nicex,*opt), '--')

path = 'C:\\Users\\nguyen89\Documents\Python Codes\Data analysis\RB data\Augustus18_QubitA_interleavedRB.txt'
data = np.genfromtxt(path)
cliffNum = [  2.,   4.,   6.,   8.,  16.,  24.,  32.,  40.,  48.,  56.,  64.,  80.,  96., 112., 128., 160., 192., 224., 256.]
for idx in range(9):
    toPlot = data[:,idx]
    print (toPlot)
    plt.errorbar(cliffNum, toPlot, fmt='o', mfc='none', mew=2.0)
    guess = ([0.99, np.max(toPlot) - np.min(toPlot), np.min(toPlot), 1])
    opt, cov = curve_fit(randomized_benchmarking_1, ydata=toPlot, xdata=cliffNum, p0=guess)
    plt.plot(nicex, randomized_benchmarking_1(nicex, *opt), '--')
#
plt.xlim([0,100])
plt.ylim([0.4,1])
plt.tick_params(labelsize = 16.0)
path = 'C:\\Users\\nguyen89\Google Drive\Research\Illustration\Thesis\Chapter 6 gates\\QubitA_RB.pdf'
plt.savefig(path, dpi=300)
plt.show()