import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*

contrast_min = -0.5
contrast_max = 0.5

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1102\Two_tone_8.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\10\Data_1031\Two_tone_3.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z,cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\10\Data_1031\Two_tone_4.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\10\Data_1031\Two_tone_5.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1101\Two_tone_5.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1103\Two_tone_8.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1104\Two_tone_8.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1104\Two_tone_7.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1107\Two_tone_14.hdf5')
freq = f.getData('IQ 1 - Frequency')[0]*1e-9
current = f.getData('Yokogawa 7651 DC Source - Current')[:,0]*1e3
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

plt.xlabel('Current (mA)')
plt.ylabel('Frequency (GHz)')
plt.colorbar()
#######################################################################################

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

Na = 25
Nb = 25
B_coeff = 60

E_la=0.45170477438306156
E_ca=0.9706755677649527
E_ja=5.842362715088368
Aa=3.731768001847992e-10
offset_a=0.5369646121203071

E_lb=0.7175559802254586
E_cb=0.9963875250852217
E_jb=5.882212077372602
Ab=3.592316009223609e-10
offset_b=0.5337843950450935

J_c = 0.12

level_num = 20
current = np.linspace(-0.5,0.5,201)*1e-3
# energies = np.zeros((len(current), level_num))
# #
# flux_a = current * B_coeff * Aa * 1e-4
# phi_ext_a = (flux_a/phi_o-offset_a) * 2 * np.pi
# flux_b = current * B_coeff * Ab * 1e-4
# phi_ext_b = (flux_b/phi_o-offset_b) * 2 * np.pi
#
# a = tensor(destroy(Na), qeye(Nb))
# phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
# na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)
#
# b = tensor(qeye(Na), destroy(Nb))
# phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
# nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)
#
# for idx in range(len(current)):
#     ope_a = 1.0j * (phi_a - phi_ext_a[idx])
#     Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
#     ope_b = 1.0j * (phi_b - phi_ext_b[idx])
#     Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
#     Hc = J_c * na * nb
#     H = Ha + Hb + Hc
#     for idy in range(level_num):
#         energies[idx,idy] = H.eigenenergies()[idy]
#     print(str(round((idx + 1) / len(current) * 100, 2)) + "%")
#
# directory = 'C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results'
# fname = "Coupled_fluxonium_spectrum_AugustusXVIII_fit_20191105.txt"
# path = directory + '\\' + fname
# np.savetxt(path, energies)
##########################################
directory = 'C:\\Users\\nguyen89\Box\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_AugustusXVIII_fit_20191105.txt"
path = directory + '\\' + fname
energies =  np.genfromtxt(path)
# for idx in range(1,level_num):
#     plt.plot(current*1e3, energies[:,idx]-energies[:,0])
# for idx in range(2,level_num):
#     plt.plot(current*1e3, energies[:,idx]-energies[:,1],'--')
# for idx in range(3,level_num):
#     plt.plot(current*1e3, energies[:,idx]-energies[:,2],'--')
# plt.plot(current*1e3, 7.5-(energies[:,1]-energies[:,0]),'-.')
# plt.plot(current*1e3, 7.5-(energies[:,2]-energies[:,0]),'-.')
plt.plot(current*1e3, energies[:,2]-energies[:,0])
plt.plot(current*1e3, energies[:,3]-energies[:,0])
plt.plot(current*1e3, energies[:,4]-energies[:,0])
plt.plot(current*1e3, energies[:,5]-energies[:,0])
plt.plot(current*1e3, energies[:,6]-energies[:,0])
plt.plot(current*1e3, energies[:,7]-energies[:,0])
plt.plot(current*1e3, energies[:,8]-energies[:,0])
plt.plot(current*1e3, energies[:,2]-energies[:,1],'--')
plt.plot(current*1e3, energies[:,3]-energies[:,1],'--')
plt.plot(current*1e3, energies[:,4]-energies[:,1],'--')
plt.plot(current*1e3, energies[:,5]-energies[:,1],'--')
plt.plot(current*1e3, energies[:,6]-energies[:,1],'--')
plt.plot(current*1e3, energies[:,7]-energies[:,1],'--')
plt.plot(current*1e3, energies[:,3]-energies[:,2],'--')
plt.plot(current*1e3, energies[:,4]-energies[:,2],'--')
plt.plot(current*1e3, energies[:,5]-energies[:,2],'--')
plt.plot(current*1e3, energies[:,6]-energies[:,2],'--')
plt.plot(current*1e3, energies[:,7]-energies[:,2],'--')

plt.plot(current*1e3, energies[:,1] - energies[:,0], linestyle ='-', label=r'$|00\rangle \rightarrow |10\rangle$')
# plt.plot(current*1e3, energies[:,3] - energies[:,2], linestyle ='--', label=r'$|01\rangle \rightarrow |11\rangle$')
plt.plot(current*1e3, energies[:,2] - energies[:,0], linestyle ='-', label=r'$|00\rangle \rightarrow |01\rangle$')
# plt.plot(current*1e3, energies[:,3] - energies[:,1], linestyle ='--', label=r'$|10\rangle \rightarrow |11\rangle$')

plt.plot(current*1e3, energies[:,5] - energies[:,3], linestyle ='-', label=r'$|11\rangle \rightarrow |12\rangle$')
plt.plot(current*1e3, energies[:,7] - energies[:,3], linestyle ='-', label=r'$|11\rangle \rightarrow |21\rangle$')
plt.plot(current*1e3, energies[:,4] - energies[:,2], linestyle ='--', label=r'$|01\rangle \rightarrow |02\rangle$')
plt.plot(current*1e3, energies[:,6] - energies[:,1], linestyle ='--', label=r'$|10\rangle \rightarrow |12\rangle$')
# plt.plot(current*1e3, energies[:,6] - energies[:,2], color = 'k', linestyle ='-', label=r'$|11\rangle \rightarrow |12\rangle$')
# plt.plot(current*1e3, energies[:,5] - energies[:,2], color = 'k', linestyle ='-', label=r'$|11\rangle \rightarrow |12\rangle$')
# plt.plot(current*1e3, energies[:,1] - energies[:,0], color = 'k', linestyle ='-', label=r'$|02\rangle \rightarrow |12\rangle$')
# plt.plot(current*1e3, energies[:,1] - energies[:,0], color = 'k', linestyle ='-', label=r'$|20\rangle \rightarrow |21\rangle$')

plt.show()