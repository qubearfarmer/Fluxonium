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
[0.005907, 6.576588],
[0.063494, 6.573115],
[0.088047, 6.569642],
[0.103225, 6.562697],
[0.132242, 6.545332],
[0.155009, 6.538386],
[0.188936, 6.510603],
[0.213936, 6.482820],
[0.250988, 6.448092],
[0.275094, 6.420309],
[0.305897, 6.375161],
[0.333574, 6.333487],
[0.378216, 6.246665],
[0.413929, 6.163316],
[0.450981, 6.055656],
[0.472855, 5.975780],
[0.490712, 5.909796],
[0.506783, 5.836865],
[0.530442, 5.715314],
[0.566602, 5.482632],
[0.583566, 5.336771],
[0.599637, 5.177018],
[0.612582, 5.027684],
[0.629993, 4.798475],
[0.646063, 4.551900],
[0.656777, 4.371311],
[0.663474, 4.256706],
[0.806573, 1.310833],
[0.813483, 1.141587],
[0.818799, 1.045926],
[0.823317, 0.957624],
[0.827836, 0.876680],
[0.833683, 0.773661],
[0.838201, 0.722151],
[0.843783, 0.655925],
[0.849896, 0.619132],
[0.853617, 0.604415],
[0.857869, 0.611774],
[0.881790, 0.884039],
[0.887638, 0.987058],
[1.041262, 4.173296],
[1.054552, 4.416127],
[1.063057, 4.563297],
[1.074486, 4.747260],
[1.085117, 4.894430],
[1.094154, 4.997450],
[1.102127, 5.100469],
[1.114354, 5.232922],
[1.124719, 5.321224],
[1.134819, 5.424243],
[1.152095, 5.556697],
[1.175750, 5.696509],
[1.187445, 5.770094],
[1.199671, 5.821603]
])
clicked_data2 = np.array([
[0.145543, 11.953159],
[0.176032, 11.874170],
[0.207710, 11.760074],
[0.233052, 11.654755],
[0.269877, 11.461670],
[0.302742, 11.259809],
[0.336795, 10.961405],
[0.385103, 10.443587],
[0.407673, 10.145183],
[0.428660, 9.829226],
[0.452814, 9.460609],
[0.464693, 9.285078],
[0.478948, 9.013004],
[0.493994, 8.749706],
[0.505478, 8.539068],
[0.520128, 8.258217],
[0.528444, 8.117792],
[0.541511, 7.863271],
[0.577940, 7.213804],
[0.599322, 6.845187],
[0.618328, 6.581890],
[0.644066, 6.248380],
[0.663469, 6.028965],
[0.688019, 5.809551],
[0.945819, 5.155773],
[0.985055, 5.487893],
[1.022287, 5.807239],
[1.046631, 6.056329],
[1.058087, 6.171293],
[1.072693, 6.330966],
[1.092169, 6.592830],
[1.111357, 6.861080],
[1.129400, 7.174040],
[1.144293, 7.416742],
[1.161191, 7.723315],
[1.168923, 7.870214],
[1.186107, 8.189560],
[1.198709, 8.406715]
])
current1 = clicked_data1[:,0]*1e-3 #In A
freq1 = clicked_data1[:,1] #in GHz

current2 = clicked_data2[:,0]*1e-3 #In A
freq2 = clicked_data2[:,1] #in GHz

current = np.concatenate([current1, current2], axis = 0)
freq = np.concatenate([freq1, freq2], axis = 0)
# current = current1
# freq = freq1
# plt.plot(current*1e3, freq, 'o') #plot mA
#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 50
B_coeff = 60
E_l_guess = 0.9
E_c_guess = 1.1
E_j_guess = 4
A_guess = 220e-12  # in m^2
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
plt.plot(current*1e3, energy[:,0],'--', current*1e3, energy[:,1],'--', current[:]*1e3, energy[:,2],'--')#, current[cut:]*1e3, energy[cut:,3],'--')
plt.show()