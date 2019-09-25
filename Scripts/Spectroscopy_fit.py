import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from scipy.optimize import curve_fit
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################


#############################################################################################
clicked_data1 = np.array([
[0.501768, 6.772820],
[0.557154, 6.772820],
[0.623735, 6.766125],
[0.728614, 6.719261],
[0.783411, 6.665702],
[0.934838, 6.384516],
[1.013793, 6.156890],
[1.089212, 5.815451],
[1.172880, 5.266470],
[1.218839, 4.750964],
[1.244764, 4.396135],
[1.279527, 3.873934],
[1.309577, 3.331648],
[1.355536, 2.501482],
[1.370266, 2.220297],
[1.351447, 2.589673],
[1.360017, 2.413149],
[1.369498, 2.236625],
[1.415627, 1.421900],
[1.429301, 1.231798],
[1.459932, 1.017932],
[1.491658, 1.218219],
[1.504785, 1.415111],
[1.543986, 2.087259],
[1.569694, 2.582884]
])
clicked_data2 = np.array([
[1.222984, 6.967775],
[1.273992, 6.378753],
[1.309476, 6.002433],
[1.418884, 4.906198],
[1.500202, 4.889836]
])
current1 = clicked_data1[:,0]*1e-3 #In A
freq1 = clicked_data1[:,1] #in GHz

current2 = clicked_data2[:,0]*1e-3 #In A
freq2 = clicked_data2[:,1] #in GHz
# current2 =[]
# freq2 = []

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
B_coeff = 45
E_l_guess = 1.2
E_c_guess = 1.0
E_j_guess = 5
A_guess = 250e-12  # in m^2
offset_guess = 0.3

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
E_l,E_c,E_j,A,offset = E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess
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
    # energy[idx, 2] = H.eigenenergies()[3] - H.eigenenergies()[0]
    energy[idx, 2] = H.eigenenergies()[2] - H.eigenenergies()[1]

cut = 400
plt.plot(current*1e3, energy[:,0],'--')
plt.plot(current*1e3, energy[:,1],'--')
plt.show()