import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from scipy.optimize import curve_fit

#####################################################################################
######################################Data###########################################
#####################################################################################
contrast_min = -2
contrast_max = 2
directory = 'G:\Projects\Fluxonium\Data\Augustus II\Two_tone'

measurement = '101318_Two_tone_spec_YOKO_0to1mA_Cav_7.56063GHz&-10dBm_QuBit3to6.3GHz&15dBm'
path = directory + '\\' + measurement
#
# #Read data
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
#
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '101318_Two_tone_spec_YOKO_0to-0.5mA_Cav_7.56063GHz&-10dBm_QuBit3to6.3GHz&15dBm'
path = directory + '\\' + measurement
#
# #Read data
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
#
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '101518_Two_tone_spec_YOKO_-0.25to0.5mA_Cav_7.5605GHz&-10dBm_QuBit3to7.3GHz&10dBm'
path = directory + '\\' + measurement
#
# #Read data
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
#
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '101618_Two_tone_spec_YOKO_0.25to0mA_Cav_7.5605GHz&-10dBm_QuBit6to7GHz&-20dBm'
path = directory + '\\' + measurement
#
# #Read data
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
#
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '101618_Two_tone_spec_YOKO_0.225to0mA_Cav_7.5605GHz&-10dBm_QuBit6to7GHz&-10dBm'
path = directory + '\\' + measurement
#
# #Read data
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
#
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

#############################################################################################
clicked_data1 = np.array([
[0.039008, 3.904501],
[0.164287, 3.913173],
[0.382516, 3.774419],
[0.507795, 3.670353],
[0.633074, 3.488237],
[0.709859, 3.314794],
[0.794725, 3.115334],
[0.859385, 2.941891],
[0.988706, 2.560316],
[1.069531, 2.291479],
[1.105903, 2.135380],
[1.219058, 1.762477],
[1.287760, 1.554345],
[1.340296, 1.380901],
[1.400915, 1.216130],
[1.429204, 1.146753],
[1.464229, 1.099582],
[1.516765, 1.060269],
[1.608367, 1.117054],
[1.659557, 1.208783],
[1.740382, 1.422818],
[1.799654, 1.606277],
[1.849496, 1.789736],
[1.908768, 1.995035],
[1.976123, 2.226543]
])
clicked_data2 = np.array([
[1.296059, 3.995099],
[1.361065, 3.919282],
[1.443676, 3.830829],
[1.580460, 3.792920],
[1.629215, 3.835041],
[1.692866, 3.894009],
[1.759227, 3.978251]
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

N = 30
B_coeff = 30
E_l_guess = 0.5
E_c_guess = 1.2
E_j_guess = 2.5
A_guess = 200e-12  # in m^2
offset_guess = 0

guess = ([E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess])

def trans_energy(current, E_l, E_c, E_j, A, offset):
    energy1 = np.zeros(len(current1))
    energy2 = np.zeros(len(current2))

    flux1 = current1 * B_coeff * A * 1e-4
    phi_ext1 = (flux1/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current1)):
        ope = 1.0j * (phi - phi_ext1[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy1[idx] = H.eigenenergies()[1] - H.eigenenergies()[0]

    flux2 = current2 * B_coeff * A * 1e-4
    phi_ext2 = (flux2 / phi_o - offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current2)):
        ope = 1.0j * (phi - phi_ext2[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy2[idx] = H.eigenenergies()[2] - H.eigenenergies()[0]

    return np.concatenate([energy1, energy2], axis=0)
    # return energy1

opt, cov = curve_fit(trans_energy, current, freq, guess)
E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit = opt
parameters_fit = {"E_l":E_l_fit, "E_c":E_c_fit, "E_j":E_j_fit, "A":A_fit,"offset":offset_fit}
for x, y in parameters_fit.items():
  print("{}={}".format(x, y))
# print ('E_l=' + str(E_l_fit) + ', E_c=' + str(E_c_fit) + ', E_j=' + str(E_j_fit) +
#        '\n' + 'A=' + str(A_fit) + ', offset='+ str(offset_fit))

############################################################################################################
# E_l,E_c,E_j,A,offset = E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess
E_l,E_c,E_j,A,offset = E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit
current = np.linspace(0,2,101)*1e-3
energy = np.zeros((len(current),10))

flux = current * B_coeff * A * 1e-4
phi_ext = (flux/phi_o-offset) * 2 * np.pi
a = tensor(destroy(N))
phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
for idx in range(len(current)):
    ope = 1.0j * (phi - phi_ext[idx])
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
    energy[idx,0] = H.eigenenergies()[1] - H.eigenenergies()[0]
    energy[idx,1] = H.eigenenergies()[2] - H.eigenenergies()[0]
    energy[idx, 2] = H.eigenenergies()[3] - H.eigenenergies()[0]
    energy[idx, 3] = H.eigenenergies()[2] - H.eigenenergies()[1]

cut = 400
plt.plot(current*1e3, energy[:,0],'--', current*1e3, energy[:,1],'--', current[cut:]*1e3, energy[cut:,2],'--', current[cut:]*1e3, energy[cut:,3],'--')
plt.show()