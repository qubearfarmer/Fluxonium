import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('seaborn-paper')
#gaussian pulse
# fname = 'Z:\Projects\Fluxonium\Data\Augustus 18\Oscilloscope data\\001.csv'
# data = np.genfromtxt(fname, skip_header=1, delimiter = ',')
# time = data[:,0]- data[0,0]
# voltage = data[:,1]
# plt.plot(time*1e9, voltage)

#gaussian pulse, I
fname = 'Z:\Projects\Fluxonium\Data\Augustus 18\Oscilloscope data\\002.csv'
data = np.genfromtxt(fname, skip_header=1, delimiter = ',')
time = data[:,0] - data[0,0]
voltage = data[:,1]
plt.plot(time*1e9, voltage, linewidth = 2.0, label = 'I')

#gaussian pulse, Q
fname = 'Z:\Projects\Fluxonium\Data\Augustus 18\Oscilloscope data\\021.csv'
data = np.genfromtxt(fname, skip_header=1, delimiter = ',')
time = data[:,0] - data[0,0]
voltage = data[:,1]
plt.plot(time*1e9, voltage, linewidth = 2.0, label = 'Q')

# fname = 'Z:\Projects\Fluxonium\Data\Augustus 18\Oscilloscope data\\02D01.csv'
# data = np.genfromtxt(fname, skip_header=1, delimiter = ',')
# time = data[:,0]- data[0,0]
# voltage = data[:,1]
# plt.plot(time*1e9, voltage, linewidth = 2.0)


# fname = 'Z:\Projects\Fluxonium\Data\Augustus 18\Oscilloscope data\\02D.csv'
# data = np.genfromtxt(fname, skip_header=1, delimiter = ',')
# time = data[:,0]- data[0,0]
# voltage = data[:,1]
# plt.plot(time*1e9, voltage, linewidth = 2.0)

# fname = 'Z:\Projects\Fluxonium\Data\Augustus 18\Oscilloscope data\\02DE.csv'
# data = np.genfromtxt(fname, skip_header=1, delimiter = ',')
# time = data[:,0]- data[0,0]
# voltage = data[:,1]
# plt.plot(time*1e9, voltage, linewidth = 2.0)


plt.show()