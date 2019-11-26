import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*
from scipy.optimize import curve_fit

contrast_max = 0.5
contrast_min= -0.5
f = Labber.LogFile('G:\Projects\Spin Chain\\2019\\11\Data_1116\\TwoTone_FluxSweep_1 GHz-8.8 GHz_0.85mA-0.43mA_-6 dBm.hdf5')

# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
f_start = f.getData('VNA_4Port - Start frequency')
f_stop = f.getData('VNA_4Port - Stop frequency')
pts = f.getData('VNA_4Port - # of points')
freq = np.linspace(f_start[0][0], f_stop[0][0], num= int(pts[0][0]))*1e-9
current = f.getData('Yoko-Roma - Current')*1e3
# print(freq)
# print(current[0])

signal = f.getData('VNA_4Port - S21')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

#############################################################################################
clicked_data1 = np.array([
[0.439315, 7.464437],
[0.468387, 7.112186],
[0.494919, 6.322291],
[0.511290, 5.607116],
[0.522581, 5.041381],
[0.730887, 4.464971],
[0.746129, 5.222843],
[0.771532, 6.386337],
[0.810202, 7.293649]
])
clicked_data2 = np.array([
[0.504798, 6.439708],
[0.517500, 5.767230],
[0.529355, 5.105426],
[0.734274, 5.052055],
[0.745847, 5.681836],
[0.759960, 6.386337]
])

current1 = clicked_data1[:,0]*1e-3 #In A
freq1 = clicked_data1[:,1] #in GHz

current2 = clicked_data2[:,0]*1e-3 #In A
freq2 = clicked_data2[:,1] #in GHz

current = np.concatenate([current1, current2], axis = 0)
freq = np.concatenate([freq1, freq2], axis = 0)
plt.plot(current*1e3, freq,'.')
#
# #############################################################
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

Na = 20
Nb = 20
E_la = 1.55
E_ca = 2
E_ja = 12.5
E_lb = 1.63
E_cb = 1.219
E_jb = 7.61
J_l = 0.002
I_o = 1.023e-3
offset = (0.633e-3-I_o/2)/I_o

guess =([E_la, E_ca, E_ja, E_lb, E_cb, E_jb, J_l])
def trans_energy(current, E_la, E_ca, E_ja, E_lb, E_cb, E_jb, J_l):
    energy1 = np.zeros(len(current1))
    energy2 = np.zeros(len(current2))

    flux1 = current1*phi_o/I_o
    phi_ext1 = (flux1/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(Na), qeye(Nb))
    phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)
    b = tensor(qeye(Na), destroy(Nb))
    phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
    nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current1)):
        ope_a = 1.0j * (phi_a - phi_ext1[idx])
        Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
        ope_b = 1.0j * (phi_b - phi_ext1[idx])
        Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
        Hc = J_l * phi_a * phi_b
        H = Ha + Hb + Hc
        energy1[idx] = H.eigenenergies()[1] - H.eigenenergies()[0]

    flux2 = current2 * phi_o / I_o
    phi_ext2 = (flux2 / phi_o - offset) * 2 * np.pi
    a = tensor(destroy(Na), qeye(Nb))
    phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)
    b = tensor(qeye(Na), destroy(Nb))
    phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
    nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current2)):
        ope_a = 1.0j * (phi_a - phi_ext2[idx])
        Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
        ope_b = 1.0j * (phi_b - phi_ext2[idx])
        Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
        Hc = J_l * phi_a * phi_b
        H = Ha + Hb + Hc
        energy2[idx] = H.eigenenergies()[2] - H.eigenenergies()[0]

    return np.concatenate([energy1, energy2], axis=0)

# opt, cov = curve_fit(trans_energy, current, freq, guess)
# E_la_fit, E_ca_fit, E_ja_fit, E_lb_fit, E_cb_fit, E_jb_fit, J_l_fit = opt
# parameters_fit = {"E_la":E_l_fit, "E_ca":E_c_fit, "E_ja":E_j_fit, "E_lb":E_l_fit, "E_cb":E_c_fit, "E_jb":E_j_fit, "J_l":J_l_fit}
# for x, y in parameters_fit.items():
#   print("{}={}".format(x, y))
# ###################################################################################
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

# E_la, E_ca, E_ja, E_lb, E_cb, E_jb, J_l = E_la_fit, E_ca_fit, E_ja_fit, E_lb_fit, E_cb_fit, E_jb_fit, J_l_fit
level_num = 20
current = np.linspace(0.43,0.85,201)*1e-3
energies = np.zeros((len(current), level_num))

flux = current*phi_o/I_o
phi_ext = (flux/phi_o-offset) * 2 * np.pi
a = tensor(destroy(Na), qeye(Nb))
phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
na = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)
b = tensor(qeye(Na), destroy(Nb))
phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
nb = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)

for idx in range(len(current)):
    ope_a = 1.0j * (phi_a - phi_ext[idx])
    Ha = 4.0 * E_ca * na ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())
    ope_b = 1.0j * (phi_b - phi_ext[idx])
    Hb = 4.0 * E_cb * nb ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())
    Hc = J_l * phi_a * phi_b
    H = Ha + Hb + Hc
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]
for idy in range(1,5):
    plt.plot(current*1e3, energies[:,idy]-energies[:,0])
plt.ylim([0,9])
plt.show()