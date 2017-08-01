#Spectrum with fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from qutip import*
plt.rc('font', family='serif')
rc('text', usetex=False)
fig=plt.figure(figsize=(14, 9))
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
'''
#####################################################################################################################################################################################
#####################################################################################################################################################################################
#Plasmon line scan
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_0to20mA_currentMode_qubit_n30dBm_cav_1dBm_avg50K_pulse25'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
Current = np.genfromtxt(path_cur, delimiter =',')*1e3
#Voltage = np.linspace(0,3,3000)
for idx in range(len(Current)-1):
    if (idx%10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx+11].transpose()
        I = Current[idx:idx+11]
        X, Y = np.meshgrid(I, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin =-5, vmax = 0)
########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_20to30mA_currentMode_qubit_n5dBm_cav_5dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
Current = np.genfromtxt(path_cur, delimiter =',')*1e3
#Voltage = np.linspace(0,3,3000)
for idx in range(len(Current)-1):
    if (idx%10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx+11].transpose()
        I = Current[idx:idx+11]
        X, Y = np.meshgrid(I, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin =-5, vmax = 0)
########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_30to32mA_currentMode_qubit_n5dBm_cav_5dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
Current = np.genfromtxt(path_cur, delimiter =',')*1e3
#Voltage = np.linspace(0,3,3000)
for idx in range(len(Current)-1):
    if (idx%10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx+11].transpose()
        I = Current[idx:idx+11]
        X, Y = np.meshgrid(I, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin =-5, vmax = 0)
########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_32to39mA_currentMode_qubit_n5dBm_cav_5dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
Current = np.genfromtxt(path_cur, delimiter =',')*1e3
#Voltage = np.linspace(0,3,3000)
for idx in range(len(Current)-1):
    if (idx%10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx+11].transpose()
        I = Current[idx:idx+11]
        X, Y = np.meshgrid(I, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin =-5, vmax = 0)
########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_39to50mA_currentMode_qubit_0dBm_cav_1dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
Current = np.genfromtxt(path_cur, delimiter =',')*1e3
#Voltage = np.linspace(0,3,3000)
for idx in range(len(Current)-1):
    if (idx%10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx+11].transpose()
        I = Current[idx:idx+11]
        X, Y = np.meshgrid(I, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin =-6, vmax = -2)

########################################################################
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_46to48mA_currentMode_qubit_0dBm_cav_5dBm_avg20K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_cur = directory + '\\' + measurement + '_Current.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
Current = np.genfromtxt(path_cur, delimiter =',')*1e3
#Voltage = np.linspace(0,3,3000)
for idx in range(len(Current)-1):
    if (idx%10) == 0:
        f = Freq[idx]
        Z = RawData[idx:idx+11].transpose()
        I = Current[idx:idx+11]
        X, Y = np.meshgrid(I, f)
        plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin =-4, vmax = -1)
#Small scan
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_43to44mA_currentMode_qubit_2p5to3p2GHz_0dBm_cav5dBm_avg50K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_current = directory + '\\' + measurement + '_I.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
I = np.genfromtxt(path_current, delimiter =',')*1e3
Z = RawData.transpose()
X, Y = np.meshgrid(I,Freq)
plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin = -4 , vmax = -1)
########################################################################
#Small scan
directory = 'C:\Data\Fluxonium #10'
measurement = 'S21_43p15to43p85mA_currentMode_qubit_1p5to2p5GHz_0dBm_cav5dBm_avg50K_pulse(test)'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_current = directory + '\\' + measurement + '_I.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
I = np.genfromtxt(path_current, delimiter =',')*1e3
Z = RawData.transpose()
X, Y = np.meshgrid(I,Freq)
plt.pcolormesh(X, Y, Z, cmap=cm.GnBu_r, vmin = -4 , vmax = -1)
########################################################################
#Blue side band
directory = 'C:\Data\Fluxonium #10'
measurement = 'Two tune spectroscopy_YOKO 43p4to43p6mA_ qubit tone 10p5to11p2GHz_5dBm_Cav_10p304GHz_8dBm_pulse 34us duty2_avg5K'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_current = directory + '\\' + measurement + '_I.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
I = (np.genfromtxt(path_current, delimiter =',')-0.00003)*1e3
Z = RawData.transpose()
X, Y = np.meshgrid(I,Freq)
plt.pcolormesh(X, Y, Z, cmap=cm.RdBu_r, vmin = -5 , vmax = 5)
########################################################################
#Red side band
directory = 'C:\Data\Fluxonium #10'
measurement = 'Two tune spectroscopy_YOKO 43p4to43p6mA_ qubit tone 8p5to10p2GHz_5dBm_Cav_10p304GHz_8dBm_pulse 34us duty2_avg5K'
path_data = directory + '\\' + measurement + '_Phase.csv'
path_freq = directory + '\\' + measurement + '_Freq.csv'
path_current = directory + '\\' + measurement + '_I.csv'

RawData = np.genfromtxt(path_data, delimiter =',')
Freq = np.genfromtxt(path_freq, delimiter =',')/1e9
I = (np.genfromtxt(path_current, delimiter =',')-0.00003)*1e3
Z = RawData.transpose()
X, Y = np.meshgrid(I,Freq)
plt.pcolormesh(X, Y, Z, cmap=cm.RdBu_r, vmin = -5 , vmax = 5)


#####################################################################################################################################################################################
#####################################################################################################################################################################################
#Plotting data taken with new software
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_41to43mA_Qubit_3to4GHz_5dBm_Cav_10.3039GHz_8dBm_IF_0.05GHz_measTime_500ns_avg_50000'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1::]-0.04
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::,0] #phase is recorded in rad
phase = phase#
mag = data[1::,0]

# plt.figure(1)
Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    Z[idx,:] = temp - np.average(temp)
Z = Z*180/(np.pi)
X,Y = np.meshgrid(current,freq[0:len(freq)/2+2])
Z1= Z.transpose()[0:len(freq)/2+2]
plt.pcolormesh(X,Y,Z1, cmap= 'GnBu_r', vmin = -4, vmax=-1, alpha = 1)

X,Y = np.meshgrid(current,freq[len(freq)/2+2:len(freq)-1])
Z2= Z.transpose()[len(freq)/2+2:len(freq)-1]
plt.pcolormesh(X,Y,Z2, cmap= 'GnBu_r', vmin = -4, vmax=-1, alpha = 1)

########################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_38.1to40mA_Qubit_3.5to5GHz_5dBm_Cav_10.3039GHz_8dBm_IF_0.05GHz_measTime_500ns_avg_25000'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1::]-0.04
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::,0] #phase is recorded in rad
phase = phase#
mag = data[1::,0]

Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    Z[idx,:] = temp - np.average(temp)
Z = Z*180/(np.pi)
for idx in range(len(current)):
    if current[idx] in [39.26, 39.29, 39.30, 39.31, 39.32, 39.43, 39.44]:
        Z[idx,:] = 0
Z = Z.transpose()[1:len(freq)-1]
X,Y = np.meshgrid(current,freq[1:len(freq)-1])
# plt.figure(1)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = -4, vmax=-1, alpha = 1)

#####################################################################
# high power scan
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_38to40mA_Qubit_3.5to5GHz_10dBm_Cav_10.3045GHz_5dBm_IF_0.05GHz_measTime_500ns_avg_50000'
path = directory + '\\' + measurement

# Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1::]-0.04
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::,0] #phase is recorded in rad
phase = phase#
mag = data[1::,0]

Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    Z[idx,:] = temp - np.average(temp)
Z = Z*180/(np.pi)
#Delete some flux data
for idx in range(len(current)):
    if current[idx] in [39.26, 39.29, 39.30, 39.31, 39.32, 39.44]:
        Z[idx,:] = 0
Z = Z.transpose()[1:len(freq)-1]
X,Y = np.meshgrid(current,freq[1:len(freq)-1])
# plt.figure(1)
# plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = -4, vmax=-1, alpha = 1)

#####################################################################
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_38.56to38.66mA_Qubit_4.2to5.1GHz_-6dBm_Cav_10.3045GHz_5dBm_IF_0.05GHz_measTime_500ns_avg_20000'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1:-1] - 0.04
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::,0] #phase is recorded in rad
phase = phase#
mag = data[1::,0]

Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    Z[idx,:] = temp - np.average(temp)
Z = Z*180/(np.pi)
Z = Z.transpose()[1:len(freq)-1]
X,Y = np.meshgrid(current,freq[1:len(freq)-1])
plt.figure(1)
# plt.pcolormesh(X,Y,Z, cmap= 'Reds_r', vmin = -4, vmax=-0.5, alpha = 0.2)
#################################################################################################
#Fine scan
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_New software'
measurement = 'Two_tone_spec_YOKO_38.56to38.66mA_Qubit_4.2to5.1GHz_-6dBm_Cav_10.3045GHz_5dBm_IF_0.05GHz_measTime_500ns_avg_20000'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURR.dat')
current = current[1:-1] - 0.037
freq = np.genfromtxt(path + '_FREQ.dat')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.dat')
phase = data[1::,0] #phase is recorded in rad
phase = phase#
mag = data[1::,0]

Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    Z[idx,:] = temp - np.average(temp)
Z = Z*180/(np.pi)
Z = Z.transpose()[1:len(freq)-1]
X,Y = np.meshgrid(current,freq[1:len(freq)-1])
plt.figure(1)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = -4, vmax=-1)
'''
#####################################################################################################################################################################################
#####################################################################################################################################################################################
#Simulation
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

#Qubit and computation parameters
N = 50
E_l = 0.722729827116
E_c = 0.552669197076
E_j_sum = 17.61374383
A_j = 4.76321410213e-12
A_c = 1.50075181762e-10
d = 0.125005274368
beta_squid = 0.129912406349
beta_ext = -0.356925557542

B_coeff = 60
level_num = 5
current = np.linspace(0.038, 0.047, 901)
energies = np.zeros((len(current),level_num))

for idx, curr in enumerate(current):
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

#Plot transition energies
for idx in range(1, 3):
    plt.plot(current*1e3, energies[:,idx]-energies[:,0], linestyle ='--',dashes=(10,10), linewidth = 1, color ='black')
# for idx in range(2, 5):
#     plt.plot(current*1e3, (energies[:,idx]-energies[:,0])/2, linestyle ='--', linewidth = 1, color ='red')

plt.tick_params(labelsize = 24)

#Full spectrum
plt.xlim([38.1,46.5])
plt.ylim([2,5])
plt.xticks(np.linspace(39,46,8))


#Anticrossing inset 1
# plt.xlim([38.523, 38.606])
# plt.ylim([4.2,5.1])
# plt.xticks([])
# plt.yticks([])

#Anticrossing inset 2
# plt.ylim([3.1, 4.0])
# plt.xlim([41.3,42.1])
# plt.xticks([])
# plt.yticks([])

#Zoom in, compared with high power spectrum 1
# plt.ylim([4.3,4.9])
# plt.xlim([38.1,38.85])
# plt.xticks(np.linspace(38.2,38.8,4))
# plt.yticks(np.linspace(4.3,4.9, 4))


#Zoom in, compared with high power spectrum 2
# plt.ylim([3.9,4.5])
# plt.xlim([39.2,39.95])
# plt.xticks(np.linspace(39.3,39.9,4))
# plt.yticks(np.linspace(3.9,4.5,4))

# directory = 'C:\\Users\\nguyen89\\Box Sync\Research\Paper Images'
# fname = 'Spectrum_wFit.eps'
# path = directory + '\\' + fname
# plt.savefig(path, format='eps', dpi=1000)

plt.show()