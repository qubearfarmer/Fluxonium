from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
plt.rc('font', family='serif')
rc('text', usetex=False)

fig=plt.figure(figsize=[5,5])
Ej = np.array([5.67, 5.09, 4.14, 3.58, 2.86, 2.58, 2.2])
gap = [0.12, 0.17, 0.33, 0.45, 0.755, 0.88, 1]

plt.errorbar(Ej**(0.5), gap, fmt='h', mfc='none', mew=2.0, mec='k', ecolor='k')
plt.yscale('log')
plt.tick_params(labelsize = 20.0)
plt.xticks([1.5,2.0,2.5])
# plt.xticks([2.5, 3.75, 5.0])
# plt.yticks([0.2, 0.6, 1])
# plt.ylim([0.1, 1.1])
plt.show()