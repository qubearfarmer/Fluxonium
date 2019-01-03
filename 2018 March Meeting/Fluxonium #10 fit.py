import numpy as np
from matplotlib import pyplot as plt

##########################################################################################
contrast_min = -1.5
contrast_max = 0.5
plt.figure(figsize =[7,7])
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_7.5GHzCav\Two_tone_spec'
# measurement = 'Two_tone_spec_YOKO_25to26mA_Cav_7.3649GHz&-15dBm_QuBit3.5to4.5GHz&5dBm'
# path = directory + '\\' + measurement
#
# #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
# current = current[1:-1]+0.02
# freq = np.genfromtxt(path + '_FREQ.csv')
# freq = freq[1:]
# data = np.genfromtxt(path + '_PHASEMAG.csv')
# phase = data[1:,0] #phase is recorded in rad
# mag = data[1:,1]
# Z = np.zeros((len(current),len(freq)))
# for idx in range(len(current)):
#     temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
#     temp = temp*180/(np.pi)
#     # temp = mag[idx*len(freq):(idx+1)*len(freq)]
#     Z[idx,:] = temp - np.mean(temp)
#
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = 'Two_tone_spec_YOKO_26to27mA_Cav_7.365GHz&-15dBm_QuBit3to4GHz&5dBm'
# path = directory + '\\' + measurement
#
# #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
# current = current[1:-1]+0.02
# freq = np.genfromtxt(path + '_FREQ.csv')
# freq = freq[1:]
# data = np.genfromtxt(path + '_PHASEMAG.csv')
# phase = data[1:,0] #phase is recorded in rad
# mag = data[1:,1]
# Z = np.zeros((len(current),len(freq)))
# for idx in range(len(current)):
#     temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
#     temp = temp*180/(np.pi)
#     # temp = mag[idx*len(freq):(idx+1)*len(freq)]
#     Z[idx,:] = temp - np.mean(temp)
#
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = 'Two_tone_spec_YOKO_27to30mA_Cav_7.365GHz&-15dBm_QuBit2.5to3.5GHz&5dBm'
# path = directory + '\\' + measurement
#
# #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
# current = current[1:-1]
# freq = np.genfromtxt(path + '_FREQ.csv')
# freq = freq[1:]
# data = np.genfromtxt(path + '_PHASEMAG.csv')
# phase = data[1:,0] #phase is recorded in rad
# mag = data[1:,1]
# Z = np.zeros((len(current),len(freq)))
# for idx in range(len(current)):
#     temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
#     temp = temp*180/(np.pi)
#     # temp = mag[idx*len(freq):(idx+1)*len(freq)]
#     Z[idx,:] = temp - np.mean(temp)
#
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
measurement = 'Two_tone_spec_YOKO_28.3to28.8mA_Cav_7.365GHz&-15dBm_QuBit0.45to2.5GHz&10dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]-0.012
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
    Z[idx,:] = temp - np.average(temp)

X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

# measurement = 'Two_tone_spec_YOKO_28.45to28.65mA_Cav_7.3649GHz&-15dBm_QuBit0.5to1.4GHz&20dBm'
# path = directory + '\\' + measurement
#
# #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
# current = current[1:-1]-0.012
# freq = np.genfromtxt(path + '_FREQ.csv')
# freq = freq[1:]
# data = np.genfromtxt(path + '_PHASEMAG.csv')
# phase = data[1:,0] #phase is recorded in rad
# mag = data[1:,1]
# Z = np.zeros((len(current),len(freq)))
# for idx in range(len(current)):
#     temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
#     temp = temp*180/(np.pi)
#     # temp = mag[idx*len(freq):(idx+1)*len(freq)]
#     Z[idx,:] = temp - np.mean(temp)
#
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)

################################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum
#################################
N = 50
B_coeff = 95.75
# current = np.linspace(25,30,501)*1e-3
current = np.linspace(28.3,28.8,501)*1e-3
E_l=0.756225886536
E_c=0.5479349297
E_j_sum=10.9255923502
A_j=7.84436850186e-12
A_c=1.52967174326e-10
d=0.203028399017
offset_squid=0.54046638981
offset_ext=0.736486271418

# def trans_energy(current, E_l, E_c, E_j_sum, d, A_j, A_c, offset_squid, offset_ext):
#     energy1 = np.zeros(len(current))
#     energy2 = np.zeros(len(current))
#     energy3 = np.zeros(len(current))
#     a = tensor(destroy(N))
#     E_j1 = 0.5 * E_j_sum * (1 + d)
#     E_j2 = 0.5 * E_j_sum * (1 - d)
#     phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
#     na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
#     flux_squid = current * B_coeff * A_j * 1e-4
#     flux_ext = current * B_coeff * A_c * 1e-4
#     phi_squid = 2.0 * np.pi * (flux_squid / phi_o - offset_squid)
#     phi_ext = 2.0 * np.pi * (flux_ext / phi_o - offset_ext)
#     for idx in range(len(current)):
#         ope1 = 1.0j * (phi_ext[idx] - phi)
#         ope2 = 1.0j * (phi + phi_squid[idx] - phi_ext[idx])
#         H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * (phi) ** 2.0 - 0.5 * E_j1 * (ope1.expm() + (-ope1).expm()) - 0.5 * E_j2 * (ope2.expm() + (-ope2).expm())
#         energy1[idx] = H.eigenenergies()[1] - H.eigenenergies()[0]
#         energy2[idx] = H.eigenenergies()[2] - H.eigenenergies()[0]
#         energy3[idx] = H.eigenenergies()[2] - H.eigenenergies()[1]
#     return energy1,energy2,energy3
# plt.plot(current*1e3+0.005, trans_energy(current, E_l, E_c, E_j_sum, d, A_j, A_c, offset_squid, offset_ext)[0],linestyle ='--',dashes=(10,10), linewidth = 0.5, color ='black')
# plt.plot(current*1e3+0.005, trans_energy(current, E_l, E_c, E_j_sum, d, A_j, A_c, offset_squid, offset_ext)[1],linestyle ='--',dashes=(10,10), linewidth = 0.5, color ='black')
# plt.plot(current*1e3+0.005, trans_energy(current, E_l, E_c, E_j_sum, d, A_j, A_c, offset_squid, offset_ext)[2],linestyle ='--',dashes=(10,10), linewidth = 0.5, color ='black')



# plt.ylim([0.0,4.5])
# plt.xlim([25,30])
# plt.yticks([0.5,0.75,1])
# # plt.xticks([25,26,27,28,29,30])
# plt.xticks([28.5, 28.55, 28.6])
# plt.tick_params(labelsize=20.0)
plt.colorbar()
plt.show()