from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import*
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################
contrast_min = -0.5
contrast_max = 1

directory = 'G:\Projects\Fluxonium\Data\\Fluxonium #28\\Two_tone'

measurement = '031818_Two_tone_spec_YOKO_0to2mA_Cav_7.3379GHz&-10dBm_QuBit2to6GHz&10dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3  b
current = current[1:-1]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1:]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1:,0] #phase is recorded in rad
mag = data[1:,1]
Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
plt.figure(1)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '032018_Two_tone_spec_YOKO_1.286to1mA_Cav_7.3379GHz&-10dBm_QuBit0.3to2GHz&25dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3  b
current = current[1:-1]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1:]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1:,0] #phase is recorded in rad
mag = data[1:,1]
Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
plt.figure(1)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '032018_Two_tone_spec_YOKO_1.258to1mA_Cav_7.3379GHz&-10dBm_QuBit0.3to2GHz&25dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3  b
current = current[1:-1]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1:]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1:,0] #phase is recorded in rad
mag = data[1:,1]
Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
plt.figure(1)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

clicked_data = np.array([
[0.004321, 5.569823],
[0.030249, 5.569823],
[0.065683, 5.577766],
[0.112352, 5.577766],
[0.162478, 5.577766],
[0.216062, 5.561881],
[0.278288, 5.522167],
[0.323228, 5.522167],
[0.391504, 5.498339],
[0.440766, 5.466568],
[0.485707, 5.418912],
[0.554847, 5.363313],
[0.616208, 5.283886],
[0.673249, 5.196516],
[0.744117, 5.069433],
[0.797700, 4.918522],
[0.851284, 4.640527],
[0.894496, 4.291048],
[0.929066, 3.933627],
[0.969685, 3.512664],
[0.991292, 3.194956],
[1.006848, 3.004331],
[1.014626, 2.893133],
[1.033640, 2.678680],
[1.151503, 1.096891],
[1.158266, 1.010229],
[1.195414, 0.549286],
[1.205225, 0.453832],
[1.208940, 0.418664],
[1.214560, 0.384753],
[1.218085, 0.367170],
[1.221323, 0.350842],
[1.226657, 0.335770],
[1.231801, 0.335770],
[1.238088, 0.357122],
[1.246756, 0.417409],
[1.251328, 0.462624],
[1.277427, 0.757778],
[1.284190, 0.848208]
])

current = clicked_data[:,0]*1e-3 #In A
freq = clicked_data[:,1] #in GHz

plt.plot(current*1e3, freq, 'o') #plot mA

#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 50
B_coeff = 30
E_l_guess = 0.9
E_c_guess = 0.85
E_j_guess = 4.8
A_guess = 280e-12  # in m^2
offset_guess = -0.00

guess = ([E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess])

def trans_energy(current, E_l, E_c, E_j, A, offset):
    energy = np.zeros(len(current))
    flux = current * B_coeff * A * 1e-4
    phi_ext = (flux/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current)):
        ope = 1.0j * (phi - phi_ext[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy[idx] = H.eigenenergies()[1] - H.eigenenergies()[0]
    return energy

opt, cov = curve_fit(trans_energy, current, freq, guess)
E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit = opt
print ('E_l=' + str(E_l_fit) + ', E_c=' + str(E_c_fit) + ', E_j=' + str(E_j_fit) +
       '\n' + 'A=' + str(A_fit) + ', offset='+ str(offset_fit))

current_nice = np.linspace(0.0, 1.5, 151)*1e-3 #In A
# plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess))
plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit))
# plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit)+7.345)

def trans_energy(current, E_l, E_c, E_j, A, offset):
    energy = np.zeros(len(current))
    flux = current * B_coeff * A * 1e-4
    phi_ext = (flux/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25)  / np.sqrt(2.0)
    for idx in range(len(current)):
        ope = 1.0j * (phi - phi_ext[idx])
        H = 4.0 * E_c * na **  2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy[idx] = H.eigenenergies()[2] - H.eigenenergies()[0]
    return energy
# plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess))
plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit))
plt.show()