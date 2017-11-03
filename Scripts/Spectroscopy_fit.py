from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import*
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################
contrast_min = -0.3
contrast_max = 0.3
directory = 'G:\Projects\BlueFors3\Data\\2017\\11\Data_1102'
measurement = 'twotone_42.hdf5'
path = directory + '\\' + measurement

#Read data
fuck = h5py.File(path, 'r')
data = (np.array((fuck.get('Data').get('Data'))))

freq = data[:,0,0]/1e9
current = data[0,1,:]*1e3
I = data[:,2,:]
Q = data[:,3,:]
phase = np.arctan2(Q,I)
X,Y = np.meshgrid(current,freq)
phase -=  np.mean(phase, axis=0)
Z = phase
plt.pcolormesh(X,Y,Z, cmap = 'GnBu', vmin = contrast_min, vmax = contrast_max)
plt.colorbar()

clicked_data = np.array([
[0.000191, 2597972972.972973],
[0.000196, 2609667359.667359],
[0.000201, 2609667359.667359],
[0.000206, 2603820166.320167],
[0.000213, 2597972972.972973],
[0.000218, 2580431392.931393],
[0.000224, 2574584199.584200],
[0.000229, 2551195426.195426],
[0.000234, 2521959459.459459],
[0.000238, 2516112266.112266],
[0.000241, 2492723492.723493],
[0.000247, 2457640332.640332],
[0.000254, 2434251559.251559],
[0.000262, 2358238045.738046],
[0.000289, 2200363825.363825],
[0.000293, 2176975051.975052],
[0.000297, 2136044698.544698],
[0.000318, 2001559251.559252],
[0.000322, 1972323284.823285],
[0.000327, 1931392931.392931],
[0.000333, 1902156964.656965],
[0.000340, 1843685031.185031],
[0.000358, 1750129937.629938],
[0.000408, 1481159043.659044],
[0.000421, 1405145530.145530],
[0.000433, 1370062370.062370],
[0.000442, 1346673596.673597],
[0.000447, 1340826403.326403],
[0.000456, 1317437629.937630],
[0.000466, 1288201663.201663],
[0.000476, 1282354469.854470],
[0.000484, 1294048856.548856],
[0.000488, 1294048856.548856]
])

current = clicked_data[:,0] #In A
freq = clicked_data[:,1]/1e9 #in GHz

plt.plot(current*1e3, freq, 'o') #plot mA

#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 50
B_coeff = 200
E_l_guess = 0.15
E_c_guess = 5
E_j_guess = 5
A_guess = 180e-12  # in m^2
offset_guess = 0.33

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

current_nice = np.linspace(0, 0.6, 101)*1e-3 #In A
plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess))
plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit))

plt.show()