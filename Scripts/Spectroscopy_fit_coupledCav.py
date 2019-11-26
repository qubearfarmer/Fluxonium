from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import*
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################
directory = "G:\Projects\Fluxonium\Data\Fluxonium waveguide 1\One_tone"
fname = '121118_One_tone_spec_3to0mA_7.05to7.1GHz_-20dBm'
path = directory + '\\' + fname

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')
current = current[1:-1]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1::]
phasemag = np.genfromtxt(path + '_PHASEMAG.csv')
phase = phasemag[1::,0] #phase is recorded in rad
phase = np.unwrap(phase)*180/(np.pi)
mag = phasemag[1::,1]
magdB = 10*np.log10(mag)
Z_mag = np.zeros((len(current), len(freq)))
Z_phase = np.zeros((len(current), len(freq)-1))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    delay = (temp[-1]-temp[0]) / (freq[-1]-freq[0])
    temp = temp - freq*delay
    # Z_phase[idx,:] = temp - np.min(temp)
    Z_phase[idx,:] = np.diff(temp - np.min(temp))
    temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z_mag[idx,:] = temp - np.min(temp)

X,Y = np.meshgrid(current, freq)
plt.figure(1)
plt.pcolormesh(X,Y+0.1,Z_mag.transpose(), cmap= 'GnBu', vmax=.002, vmin = -.000)

fname = '121218_One_tone_spec_0.9to1.15mA_7.0to7.2GHz_-20dBm'
path = directory + '\\' + fname

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')
current = current[1:-1]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1::]
phasemag = np.genfromtxt(path + '_PHASEMAG.csv')
phase = phasemag[1::,0] #phase is recorded in rad
phase = np.unwrap(phase)*180/(np.pi)
mag = phasemag[1::,1]
magdB = 10*np.log10(mag)
Z_mag = np.zeros((len(current), len(freq)))
Z_phase = np.zeros((len(current), len(freq)-1))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    delay = (temp[-1]-temp[0]) / (freq[-1]-freq[0])
    temp = temp - freq*delay
    # Z_phase[idx,:] = temp - np.min(temp)
    Z_phase[idx,:] = np.diff(temp - np.min(temp))
    temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z_mag[idx,:] = temp - np.min(temp)

X,Y = np.meshgrid(current, freq)
plt.figure(1)
plt.pcolormesh(X,Y+0.1,Z_mag.transpose(), cmap= 'GnBu', vmax=.002, vmin = -.000)

#############################################################################################
clicked_data1 = np.array([
[0.022956, 7.181178],
[0.080686, 7.181716],
[0.156889, 7.181986],
[0.281586, 7.181986],
[0.440921, 7.181716],
[0.535598, 7.181716],
[0.646440, 7.181986],
[0.803465, 7.180908],
[0.902761, 7.179023],
[0.960491, 7.176061],
[0.985892, 7.170405],
[1.002056, 7.163672],
[1.006675, 7.155323],
[1.018221, 7.148052],
[1.022839, 7.138895],
[2.054079, 7.157144],
[2.060678, 7.167313],
[2.095876, 7.176738],
[2.179469, 7.179715],
[2.342255, 7.180707],
[2.494042, 7.181203],
[2.676627, 7.181203],
[2.797617, 7.181203]
])
clicked_data2 = np.array([
[1.009908, 7.217760],
[1.018853, 7.207484],
[1.023325, 7.196524],
[1.056869, 7.188303],
[1.144084, 7.184878],
[1.255898, 7.182481],
[1.361003, 7.182481],
[1.468344, 7.181796],
[1.568977, 7.181796],
[1.725516, 7.181796],
[1.879820, 7.183508],
[1.982688, 7.187276],
[2.036359, 7.197894]
])
current1 = clicked_data1[:,0]*1e-3 #In A
freq1 = clicked_data1[:,1] #in GHz

current2 = clicked_data2[:,0]*1e-3 #In A
freq2 = clicked_data2[:,1] #in GHz

current = np.concatenate([current1, current2], axis = 0)
freq = np.concatenate([freq1, freq2], axis = 0)
# current = current1
# freq = freq1
plt.plot(current*1e3, freq, 'o') #plot mA
#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

Na = 30
Nr = 5
B_coeff = 30
E_l = 0.5919
E_c = 1.1763
E_j = 2.0482
A = 233.7e-12  # in m^2
offset = 0.019
wr_guess = 7.18
g_guess = 0.25
guess = ([wr_guess, g_guess])

def trans_energy(current, wr, g):
    energy1 = np.zeros(len(current1))
    energy2 = np.zeros(len(current2))

    flux1 = current1 * B_coeff * A * 1e-4
    phi_ext1 = (flux1/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(Na), qeye(Nr))
    b = tensor(qeye(Na), destroy(Nr))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current1)):
        ope = 1.0j * (phi - phi_ext1[idx])
        H_f = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        H_r = wr * (b.dag() * b + 1.0 / 2)
        H_c = g * na * (b.dag() + b)
        H = H_f + H_r + H_c
        energy1[idx] = H.eigenenergies()[3] - H.eigenenergies()[0]

    flux2 = current2 * B_coeff * A * 1e-4
    phi_ext2 = (flux2 / phi_o - offset) * 2 * np.pi
    a = tensor(destroy(Na), qeye(Nr))
    b = tensor(qeye(Na), destroy(Nr))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current2)):
        ope = 1.0j * (phi - phi_ext2[idx])
        H_f = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        H_r = wr * (b.dag() * b + 1.0 / 2)
        H_c = g * na * (b.dag() + b)
        H = H_f + H_r + H_c
        energy2[idx] = H.eigenenergies()[4] - H.eigenenergies()[0]

    return np.concatenate([energy1, energy2], axis=0)
    # return energy1

opt, cov = curve_fit(trans_energy, current, freq, guess)
wr_fit, g_fit = opt
parameters_fit = {"wr" : wr_fit, "g" : g_fit}
for x, y in parameters_fit.items():
  print("{}={}".format(x, y))

############################################################################################################
# wr, g = wr_guess, g_guess
# wr, g = wr_fit, g_fit
# current = np.linspace(0,3,301)*1e-3
# energy = np.zeros((len(current),10))
#
# flux = current * B_coeff * A * 1e-4
# phi_ext = (flux/phi_o-offset) * 2 * np.pi
# a = tensor(destroy(N))
# phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
# na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
# for idx in range(len(current)):
#     ope = 1.0j * (phi - phi_ext[idx])
#     H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
#     energy[idx,0] = H.eigenenergies()[1] - H.eigenenergies()[0]
    # energy[idx,1] = H.eigenenergies()[2] - H.eigenenergies()[0]
    # energy[idx, 2] = H.eigenenergies()[3] - H.eigenenergies()[0]
    # energy[idx, 3] = H.eigenenergies()[2] - H.eigenenergies()[1]

# cut = 400
# plt.plot(current*1e3, energy[:,0],'--')#, current*1e3, energy[:,1],'--', current[cut:]*1e3, energy[cut:,2],'--', current[cut:]*1e3, energy[cut:,3],'--')
plt.show()