import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap

# plt.rc('text', usetex=True)
# plt.rc('font', family='sans-serif')
# Define file directory
root = "C:\\Users\\nguyen89"
directory = "Box\Python Codes\Fluxonium simulation results"
fname = "Relaxation_freq"
path = root + "\\" + directory + "\\" + fname

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.626e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum
plt.figure(figsize=[7, 2.5])
#######################################################################################
N = 50
E_l = 1
E_c = 0.8
E_jt = np.linspace(1.0, 6.0, 601)
level_num = 15
kB = 1.38e-23
Q1 = 0.28e6
Q2 = 0.5e6
T_diel = 20.0e-3
T_qp = 20.0e-3
alpha = 0.15

iState = 0
fState = 1
phi = 0.5
p_element = np.zeros(len(E_jt))
n_element = np.zeros(len(E_jt))
qp_element = np.zeros(len(E_jt))
gamma_cap = np.zeros(len(E_jt))
gamma_flux = np.zeros(len(E_jt))
gamma_qp = np.zeros(len(E_jt))
gamma_qp_array = np.zeros(len(E_jt))
energies = np.zeros((len(E_jt), level_num))

#######################################################################################
# for idx, E_j in enumerate(E_jt):
#     p_element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
#     n_element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2.0*np.pi, iState, fState))
#     qp_element[idx] = abs(qpem(N, E_l, E_c, E_j, phi * 2.0 * np.pi, iState, fState))
#     for idy in range(level_num):
#         energies[idx,idy] = H(N, E_l, E_c, E_j, phi*2.0*np.pi).eigenenergies()[idy]
#
# np.savetxt(path + '_energies.txt', energies)
# np.savetxt(path + '_chargeElement.txt', n_element)
# np.savetxt(path + '_fluxElement.txt', p_element)
# np.savetxt(path + '_qpElement.txt', qp_element)
#######################################################################################

energies = np.genfromtxt(path + '_energies.txt')
n_element = np.genfromtxt(path + '_chargeElement.txt')
p_element = np.genfromtxt(path + '_fluxElement.txt')
qp_element = np.genfromtxt(path + '_qpElement.txt')
w = energies[:, fState] - energies[:, iState]
thermal_factor_diel = (1 + np.exp(-h * w * 1e9 / (kB * T_diel)))
# thermal_factor_qp = (1+np.exp(-h*w*1e9/(kB*T_qp)))

for Q_cap in [Q1, Q2]:
    for idx in range(len(E_jt)):
        gamma_cap[idx] = r_cap(E_l, E_c, E_jt[idx], Q_cap * (6.0 / w[idx]) ** alpha, w[idx], p_element[idx], T_diel) * \
                         thermal_factor_diel[idx]
    plt.loglog(w, 1.0 / gamma_cap * 1e6 * p_element ** 2 / E_c, linewidth=2.0, linestyle='--', color='orange')
#
# for Q_cap in [Q1,Q2]:
#     for idx in range(len(E_jt)):
#         gamma_cap[idx] = r_cap(E_l, E_c, E_jt[idx], Q_cap*(6.0/w[idx])**alpha, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
#     plt.loglog(w, 1.0/gamma_cap*1e6*p_element**2/E_c, linewidth= 2.0, linestyle ='--', color = 'magenta')

# T_diel=50e-3
# thermal_factor_diel = (1+np.exp(-h*w*1e9/(kB*T_diel)))
# for Q_cap in [0.5e6]:
#     for idx in range(len(E_jt)):
#         gamma_cap[idx] = r_cap(E_l, E_c, E_jt[idx], Q_cap, w[idx], p_element[idx], T_diel)*thermal_factor_diel[idx]
#     plt.loglog(w, 1.0/gamma_cap *1e6*p_element**2, linewidth= 2.0, linestyle ='-')

# for x_qp in [100e-7, 5e-7]:
#     Q_qp = 1.0/x_qp
#     for idx in range(len(E_jt)):
#         gamma_qp[idx] = r_qp(E_l, E_c, E_jt[idx], Q_qp, w[idx], qp_element[idx], T_qp)*thermal_factor_qp[idx]
#     # plt.semilogy(w, 1.0/(gamma_qp)*1e6, linewidth = 2.0, linestyle='--')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap)*1e6, linewidth = 2.0, linestyle='-.', color='k')

# for x_qp in [2e-9, 5.0e-9, 10e-9]:
#     Q_qp = 1.0 / x_qp
#     for idx in range(len(E_jt)):
#         gamma_qp_array[idx] = r_qp_array(E_l, E_c, E_jt[idx], Q_qp, w[idx], p_element[idx], T_qp)*thermal_factor_qp[idx]
#     plt.semilogy(w, 1.0 / (gamma_qp_array) * 1e6, linewidth=2.0, linestyle='--', color = 'green')
#     plt.semilogy(w, 1.0/(gamma_qp+gamma_cap+gamma_qp_array)*1e6, linewidth = 2.5, linestyle='-', color ='black')

w = np.array([0.8, 0.3, 0.5, 0.6, 0.9, 0.18, 0.55, 0.32])
T1_A = np.array([
    98.6617873, 95.5015335, 95.25972457, 96.47774671,
    93.80886189, 92.02312106, 92.00390993, 104.64146104,
    98.99336332, 98.61228503, 102.34441115, 105.67657043,
    106.18142521, 105.45833176, 98.45673552, 101.61287509,
    90.98371399, 101.18348887, 94.17932204, 100.45472872,
    95.57827225, 87.7495575, 87.55386756, 111.107124,
    110.16259355, 96.63317222, 105.14646218, 98.44706896,
    95.27461215, 96.29180325, 85.49935081, 97.71202044,
    90.27759636, 100.01682325, 86.73974643, 84.43136988,
    98.1055042, 91.17735569, 82.78587454, 90.61516412,
    94.94906067, 84.61803569, 86.25987445, 87.74893379,
    88.65070431, 88.87418738, 85.41452357, 82.56292197,
    88.04836013, 88.9943133, 90.13159676, 93.47189073,
    94.10562099, 95.91581081, 93.72577438, 104.13409716,
    104.22802847, 105.29532251, 108.42431025, 104.51931133,
    97.53770833, 103.26950803, 109.55464673, 112.55972324,
    103.81139606, 104.15719508, 119.20181814, 112.83216216,
    105.01594934, 115.27026828, 100.29206182, 98.321432,
    97.11419648, 106.66273862, 91.55069718, 101.72050207,
    91.99297294, 94.54120588, 101.08177333, 102.83206654,
    106.6614754, 101.90927167, 101.63445438, 93.8476734,
    98.58202279, 90.55989684, 92.57146624, 90.27721884,
    91.98224446, 89.95205465, 103.80290478, 98.43527169,
    100.61389005, 108.9572145, 97.75521355, 92.69929367,
    106.56367157, 90.98745826, 86.6815001, 91.55099326
])
T1_B = np.array([248, 235, 216, 212, 264, 232, 280])
T1_C = np.array([193.15545423, 175.95341997, 205.77462291, 171.87092679,
                 230.49670798, 197.76209522, 264.45566489, 203.22166066,
                 213.75496903, 221.38494221, 236.76896582, 167.57477402,
                 204.10297706, 191.99384926, 174.46290858, 187.26577069,
                 192.97235653, 146.78086537, 182.84226262, 188.43005545])
T1_D = np.array([50.37214565, 57.85653646, 51.59341153, 61.08138398, 67.74016964,
                 63.62142066, 63.74178745, 58.59837256, 57.77241213, 64.54200142,
                 55.15695825, 54.17164495, 53.01450589, 63.00241295, 61.12681768,
                 56.35247025, 57.02326511, 57.30806335, 58.70431241, 57.95588348,
                 54.34713116, 56.65042738, 57.05866393, 57.33821701, 56.59472386,
                 55.24868214, 57.22034005, 56.5531287, 47.83640116, 53.86174348,
                 53.36888817, 55.26972793, 60.56961454, 59.94604976, 60.58763461,
                 65.9421445, 60.17018744, 69.98748169, 70.48702599, 68.14707314,
                 68.78317647, 55.75599514, 62.05581935, 48.70114365, 59.65148551])
T1_E = np.array([108.23094119, 83.40211799, 103.74129417, 105.18766447, 102.58817742,
                 107.18082663, 98.78535853, 71.1850189, 68.90596416, 62.28266865])
T1_F = np.array([154, 160, 220, 260, 273])
T1_G = np.array([92.68876018, 109.97439375, 96.47169406, 94.33707057,
                 80.44040539, 106.29870081, 98.80441059, 117.68108584,
                 97.94531839, 84.95991315, 95.50692121, 111.4117933,
                 90.46373398, 87.38596863, 92.38653998, 84.51528698,
                 78.61403468, 97.22743339, 100.08560086, 85.68151091,
                 83.86361, 83.98814727, 106.81873048, 72.45242161,
                 76.634694, 91.77520584, 74.6273798, 79.15591729,
                 88.9706327, 81.08243984, 77.99103525, 88.69615652,
                 92.60126195, 97.35243161, 95.03104054, 79.35624105,
                 77.08121738, 82.67266249, 80.06353891, 79.04084529,
                 93.57372984, 71.19204014, 58.3849639, 78.07248064,
                 66.98457428, 72.1317285, 68.89561441, 79.80733501,
                 70.39991118, 81.59414928])
T1_H = np.array([164.67191539, 142.24824189, 143.42348887, 138.005341,
                 152.96311531, 131.39985872, 184.40807024, 221.96529117,
                 134.03890868, 200.80851945, 156.194309, 133.29342269,
                 177.6778851, 182.07421536, 173.86673724, 214.08582172,
                 136.28949857, 197.55858918, 161.08311553, 160.02623714,
                 185.47578274, 221.55483248, 153.64188024, 169.58840324,
                 230.90571487, 163.1798721, 127.0558945, 139.88898418,
                 140.66191989, 178.04854167, 148.86637543, 162.57972445,
                 182.1330399, 134.44596237, 108.38090278, 111.36892662,
                 103.89201035, 105.98334149, 139.49624558, 137.38199679,
                 96.06559806, 97.9521803, 188.48027537, 163.7749385,
                 174.3841941, 133.55528938, 165.0890997, 178.2466385,
                 182.13058878, 157.24213999, 141.74139706
                 ])
# T1 = np.array ([110, 250, 260, 70, 100, 270, 110*0.84/1.14])
T1 = np.array([T1_A, T1_B, T1_C, T1_D, T1_E, T1_F, T1_G * 0.84 / 1.14, T1_H * 0.84 / 1.0])
T2 = np.array([160, 150, 350, 90, 140, 165, 140])
Ec_a = np.array([0.84, 0.84, 0.8, 0.83, 0.86, 0.83, 1.14, 1])
m_element = np.array([1.875, 2.243, 1.896, 2.127, 1.923, 2.645, 2.51, 2.4])
N = np.array([100, 136, 102, 196, 100, 348, 400])
tan = np.array([1.7, 1.5, 1.25, 3.1, 1.6, 2.2, 2])
x = w
y = T1 * m_element ** 2
for xe, ye in zip(x, y):
    plt.errorbar([xe] * len(ye), ye, fmt='s', mfc='none', mew=2.0, mec='blue')
plt.yscale('log')
plt.xscale('log')
# T2_lf = np.array ([28, 14, 40, 1e100, 48, 55])
# plt.yscale('log')
# plt.xscale('log')
# plt.errorbar(w,T2, fmt = 's', mfc = 'none', mew = 2.0, label=r'$T_2e$',mec='g')
# plt.errorbar(w,T2_lf, fmt = 's', mfc = 'none', mew = 2.0, label='low field T2')
# w=np.array([0.33,0.172])
# T1=np.array([120,150])
# plt.errorbar(w,T1, fmt = 's', mfc = 'none', mew = 2.0, label='low field T1', mec='red')

# plt.xticks([0.5, 1, 5])
# plt.yticks([0, 100, 200, 300, 400])
plt.tick_params(labelsize=18.0)
# plt.grid()
plt.yticks([])
plt.xticks([])
plt.legend()
plt.show()
