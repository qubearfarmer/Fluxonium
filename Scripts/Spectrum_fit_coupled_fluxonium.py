from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import*
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################

#############################################################################################
clicked_data1 = np.array([
[-0.494124, 5.921000],
[-0.382491, 5.912435],
[-0.310307, 5.792532],
[-0.217139, 4.353687],
[-0.159224, 3.368764],
[-0.014856, 0.859350],
[0.062364, 0.508203],
[0.135387, 1.767193],
[0.347742, 5.432820],
[0.373762, 5.775403],
[0.438392, 5.912435],
[0.497146, 5.921000],
[0.004324, 0.540121],
[0.035001, 0.072349],
[0.076614, 0.722033]
])
clicked_data2 = np.array([
[-0.487399, 6.232272],
[-0.366431, 6.205908],
[-0.326109, 6.179544],
[-0.296707, 5.959847],
[-0.228663, 5.863181],
[-0.204301, 5.704999],
[-0.184980, 5.300757],
[-0.069052, 2.506212],
[-0.016129, 1.188030],
[0.034274, 0.151061],
[0.077957, 1.152879],
[0.254368, 5.441363],
[0.274530, 5.775302],
[0.304772, 5.880757],
[0.372816, 6.012575],
[0.388777, 6.170756],
[0.496304, 6.232272]
])
clicked_data3 = np.array([
[-0.367915, 6.956998],
[-0.336990, 6.461750],
[-0.321945, 6.241640],
[-0.311915, 6.178752],
[-0.263438, 6.115863],
[-0.214960, 5.927197],
[-0.197408, 5.864309],
[0.239726, 5.832864],
[0.273994, 5.880031],
[0.329158, 6.115863],
[0.382651, 6.186613],
[0.397696, 6.328112],
[0.436979, 6.980581]
])
clicked_data4 = np.array([
[-0.261089, 7.351861],
[-0.242104, 6.891385],
[-0.211869, 6.349306],
[-0.188665, 6.151126],
[-0.159133, 6.057865],
[-0.145773, 5.935460],
[-0.122569, 5.766425],
[0.345729, 7.789021],
[0.284555, 6.430909],
[0.254319, 6.151126],
[0.230412, 6.063694],
[0.207912, 5.888830],
[0.181895, 5.766425]
])
clicked_data5 = np.array([
# [0.833717, 11.953594],
# [0.898281, 11.479694],
# [0.954351, 11.053185],
# [1.024579, 10.543743],
# [1.069322, 10.200166],
# [1.219973, 8.600755],
# [1.244327, 8.020228]
])
clicked_data6 = np.array([
# [0.909543, 12.219887],
# [0.959812, 11.974461],
# [1.047043, 11.352716],
# [1.071438, 11.156375]
])
clicked_data7 = np.array([
# [1.100269, 12.203525]
])
clicked_data8 = np.array([
# [1.216331, 10.747332],
# [1.211895, 10.845503],
# [1.202285, 10.960035],
# [1.163105, 11.483610],
# [1.123925, 11.990823]
])
current1 = clicked_data1[:,0]*1e-3 #In A
freq1 = clicked_data1[:,1] #in GHz

current2 = clicked_data2[:,0]*1e-3 #In A
freq2 = clicked_data2[:,1] #in GHz

current3 = clicked_data3[:,0]*1e-3 #In A
freq3 = clicked_data3[:,1] #in GHz

current4 = clicked_data4[:,0]*1e-3 #In A
freq4 = clicked_data4[:,1] #in GHz

# current5 = clicked_data5[:,0]*1e-3 #In A
# freq5 = clicked_data5[:,1] #in GHz
#
# current6 = clicked_data6[:,0]*1e-3 #In A
# freq6 = clicked_data6[:,1] #in GHz
#
# current7 = clicked_data7[:,0]*1e-3 #In A
# freq7 = clicked_data7[:,1] #in GHz
#
# current8 = clicked_data8[:,0]*1e-3 #In A
# freq8 = clicked_data8[:,1] #in GHz

current = np.concatenate([current1, current2, current3, current4], axis = 0)#, current6, current7, current8], axis = 0)
freq = np.concatenate([freq1, freq2, freq3, freq4], axis = 0)#, freq6, freq7, freq8], axis = 0)
# current = current1
# freq = freq1
#plt.plot(current*1e3, freq, 'o') #plot mA
#plt.show()
#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

Na = 25
Nb = 25
B_coeff = 60

E_la_guess=0.45170477438306156
E_ca_guess=0.9706755677649527
E_ja_guess=5.842362715088368
Aa=3.731768001847992e-10
offset_a=0.5369646121203071

E_lb_guess=0.7175559802254586
E_cb_guess=0.9963875250852217
E_jb_guess=5.882212077372602
Ab=Aa
offset_b=offset_a

J_c_guess = 0.2

guess = ([E_la_guess, E_ca_guess, E_ja_guess, E_lb_guess, E_cb_guess, E_jb_guess, J_c_guess])

def trans_energy(current,E_la, E_ca, E_ja, E_lb, E_cb, E_jb, J_c):
    energy1 = np.zeros(len(current1))
    flux1a = current1 * B_coeff * Aa * 1e-4
    phi_ext1a = (flux1a/phi_o-offset_a) * 2 * np.pi
    flux1b = current1 * B_coeff * Ab * 1e-4
    phi_ext1b = (flux1b / phi_o - offset_b) * 2 * np.pi

    a = tensor(destroy(Na), qeye(Nb))
    phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)

    b = tensor(qeye(Na), destroy(Nb))
    phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
    nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)

    for idx in range(len(current1)):
        ope_a = 1.0j * (phi_a - phi_ext1a[idx])
        Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
        ope_b = 1.0j * (phi_b - phi_ext1b[idx])
        Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
        Hc = J_c * na * nb
        H = Ha + Hb + Hc
        energy1[idx] = H.eigenenergies()[1] - H.eigenenergies()[0]

    energy2 = np.zeros(len(current2))
    flux2a = current2 * B_coeff * Aa * 1e-4
    phi_ext2a = (flux2a / phi_o - offset_a) * 2 * np.pi
    flux2b = current2 * B_coeff * Ab * 1e-4
    phi_ext2b = (flux2b / phi_o - offset_b) * 2 * np.pi

    for idx in range(len(current2)):
        ope_a = 1.0j * (phi_a - phi_ext2a[idx])
        Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
        ope_b = 1.0j * (phi_b - phi_ext2b[idx])
        Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
        Hc = J_c * na * nb
        H = Ha + Hb + Hc
        energy2[idx] = H.eigenenergies()[2] - H.eigenenergies()[0]

    energy3 = np.zeros(len(current3))
    flux3a = current3 * B_coeff * Aa * 1e-4
    phi_ext3a = (flux3a / phi_o - offset_a) * 2 * np.pi
    flux3b = current3 * B_coeff * Ab * 1e-4
    phi_ext3b = (flux3b / phi_o - offset_b) * 2 * np.pi


    for idx in range(len(current3)):
        ope_a = 1.0j * (phi_a - phi_ext3a[idx])
        Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
        ope_b = 1.0j * (phi_b - phi_ext3b[idx])
        Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
        Hc = J_c * na * nb
        H = Ha + Hb + Hc
        energy3[idx] = H.eigenenergies()[3] - H.eigenenergies()[0]

    energy4 = np.zeros(len(current4))
    flux4a = current4 * B_coeff * Aa * 1e-4
    phi_ext4a = (flux4a / phi_o - offset_a) * 2 * np.pi
    flux4b = current4 * B_coeff * Ab * 1e-4
    phi_ext4b = (flux4b / phi_o - offset_b) * 2 * np.pi


    for idx in range(len(current4)):
        ope_a = 1.0j * (phi_a - phi_ext4a[idx])
        Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
        ope_b = 1.0j * (phi_b - phi_ext4b[idx])
        Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
        Hc = J_c * na * nb
        H = Ha + Hb + Hc
        energy4[idx] = H.eigenenergies()[4] - H.eigenenergies()[0]

    # energy5 = np.zeros(len(current5))
    # flux5a = current5 * B_coeff * Aa * 1e-4
    # phi_ext5a = (flux5a / phi_o - offset_a) * 2 * np.pi
    # flux5b = current5 * B_coeff * Ab * 1e-4
    # phi_ext5b = (flux5b / phi_o - offset_b) * 2 * np.pi
    #
    #
    # for idx in range(len(current5)):
    #     ope_a = 1.0j * (phi_a - phi_ext5a[idx])
    #     Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
    #     ope_b = 1.0j * (phi_b - phi_ext5b[idx])
    #     Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
    #     Hc = J_c * na * nb
    #     H = Ha + Hb + Hc
    #     energy5[idx] = H.eigenenergies()[5] - H.eigenenergies()[0]
    #
    # energy6 = np.zeros(len(current6))
    # flux6a = current6 * B_coeff * Aa * 1e-4
    # phi_ext6a = (flux6a / phi_o - offset_a) * 2 * np.pi
    # flux6b = current6 * B_coeff * Ab * 1e-4
    # phi_ext6b = (flux6b / phi_o - offset_b) * 2 * np.pi
    #
    #
    # for idx in range(len(current6)):
    #     ope_a = 1.0j * (phi_a - phi_ext6a[idx])
    #     Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
    #     ope_b = 1.0j * (phi_b - phi_ext6b[idx])
    #     Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
    #     Hc = J_c * na * nb
    #     H = Ha + Hb + Hc
    #     energy6[idx] = H.eigenenergies()[6] - H.eigenenergies()[0]
    #
    # energy7 = np.zeros(len(current7))
    # flux7a = current7 * B_coeff * Aa * 1e-4
    # phi_ext7a = (flux7a / phi_o - offset_a) * 2 * np.pi
    # flux7b = current7 * B_coeff * Ab * 1e-4
    # phi_ext7b = (flux7b / phi_o - offset_b) * 2 * np.pi
    #
    #
    # for idx in range(len(current7)):
    #     ope_a = 1.0j * (phi_a - phi_ext7a[idx])
    #     Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
    #     ope_b = 1.0j * (phi_b - phi_ext7b[idx])
    #     Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
    #     Hc = J_c * na * nb
    #     H = Ha + Hb + Hc
    #     energy7[idx] = H.eigenenergies()[7] - H.eigenenergies()[0]
    #
    # energy8 = np.zeros(len(current8))
    # flux8a = current8 * B_coeff * Aa * 1e-4
    # phi_ext8a = (flux8a / phi_o - offset_a) * 2 * np.pi
    # flux8b = current8 * B_coeff * Ab * 1e-4
    # phi_ext8b = (flux8b / phi_o - offset_b) * 2 * np.pi
    #
    # for idx in range(len(current8)):
    #     ope_a = 1.0j * (phi_a - phi_ext8a[idx])
    #     Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
    #     ope_b = 1.0j * (phi_b - phi_ext8b[idx])
    #     Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
    #     Hc = J_c * na * nb
    #     H = Ha + Hb + Hc
    #     energy8[idx] = H.eigenenergies()[8] - H.eigenenergies()[0]

    return np.concatenate([energy1, energy2, energy3, energy4], axis=0)
    # return np.concatenate([energy1, energy2, energy3, energy4, energy5, energy6, energy7, energy8], axis=0)

opt, cov = curve_fit(trans_energy, xdata=current, ydata=freq, p0=guess)
E_la_fit, E_ca_fit, E_ja_fit, E_lb_fit, E_cb_fit, E_jb_fit, J_c_fit = opt
# print ('E_la=' + str(E_la_fit) + ', E_ca=' + str(E_ca_fit) + ', E_ja=' + str(E_ja_fit) +
#        '\n' + 'E_lb=' + str(E_lb_fit) + ', E_cb=' + str(E_cb_fit) + ', E_jb=' + str(E_jb_fit) +
#        '\n' + 'A=' + str(Aa_fit) + ', offset=' + str(offset_a_fit) + ', J_c='+ str(J_c_fit))
print (opt)
parameters_fit = {"E_la":E_la_fit, "E_ca":E_ca_fit, "E_ja":E_ja_fit,
                  "E_lb": E_lb_fit, "E_cb": E_cb_fit, "E_jb": E_jb_fit,
                  "J_c":J_c_fit}
for x, y in parameters_fit.items():
  print("{}={}".format(x, y))

############################################################################################################

E_la, E_ca, E_ja, E_lb, E_cb, E_jb, J_c = E_la_fit, E_ca_fit, E_ja_fit, E_lb_fit, E_cb_fit, E_jb_fit, J_c_fit

# Aa=Aa_guess
# offset_a=0.028043712988006286
#
# Ab=2.1086704960372626e-10
# offset_b=0.02347013794896157

# level_num = 20
# current = np.linspace(0.5,2,751)*1e-3
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
# fname = "Coupled_fluxonium_spectrum_AugustusXVI_fit_20190906.txt"
# path = directory + '\\' + fname
# np.savetxt(path, energies)


