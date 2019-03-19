import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
rc('text', usetex=False)
plt.figure(figsize = [6,6])
import h5py
from qutip import*


#######################################################################################################
# contrast_min = 0
# contrast_max = 3
#
# directory = 'G:\Projects\Fluxonium\Data\Augustus III\Two tone'
#
# measurement = '120718_Two_tone_spec_DC_YOKO_0to10mA_Cav_7.45671GHz&-20dBm_QuBit4to6GHz&0dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '120718_Two_tone_spec_DC_YOKO_0.8to10mA_Cav_7.45671GHz&-20dBm_QuBit4to6GHz&10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '120718_Two_tone_spec_DC_YOKO_6to10mA_Cav_7.45671GHz&-20dBm_QuBit4to6GHz&10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '120718_Two_tone_spec_DC_YOKO_8to16mA_Cav_7.45671GHz&-20dBm_QuBit2to6GHz&10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '120718_Two_tone_spec_DC_YOKO_14.35to10mA_Cav_7.45671GHz&-20dBm_QuBit0.3to2GHz&15dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '120718_Two_tone_spec_DC_YOKO_13.5to10.2mA_Cav_7.45671GHz&-20dBm_QuBit0.3to2GHz&25dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '120818_Two_tone_spec_DC_YOKO_14to0mA_Cav_7.45671GHz&-20dBm_QuBit6to9GHz&-10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '120818_Two_tone_spec_DC_YOKO_14to0mA_Cav_7.45671GHz&-20dBm_QuBit9to12GHz&-10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '120818_Two_tone_spec_DC_YOKO_14to0mA_Cav_7.45671GHz&-20dBm_QuBit9to12GHz&-10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '120818_Two_tone_spec_DC_YOKO_10.5to13.5mA_Cav_7.45671GHz&-20dBm_QuBit0.5to2GHz&10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '120918_Two_tone_spec_DC_YOKO_14to0mA_Cav_7.45671GHz&-20dBm_QuBit9to12GHz&5dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '120918_Two_tone_spec_DC_YOKO_11.5to0mA_Cav_7.45671GHz&-20dBm_QuBit9to12GHz&5dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '120918_Two_tone_spec_DC_YOKO_0to14mA_Cav_7.45671GHz&-20dBm_QuBit6to9GHz&-12dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '121018_Two_tone_spec_DC_YOKO_13to11mA_Cav_7.45671GHz&-20dBm_QuBit0.1to0.3GHz&10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '121018_Two_tone_spec_DC_YOKO_12to12.4mA_Cav_7.45671GHz&-20dBm_QuBit0.506to0.54GHz&10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '121018_Two_tone_spec_DC_YOKO_12.24to12.14mA_Cav_7.45671GHz&-20dBm_QuBit0.509to0.513GHz&10dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '121018_Two_tone_spec_DC_YOKO_7to0mA_Cav_7.45671GHz&-20dBm_QuBit9to12GHz&15dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
#
# measurement = '121018_Two_tone_spec_DC_YOKO_3.8to0mA_Cav_7.45671GHz&-20dBm_QuBit9to12GHz&25dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
#
# measurement = '121018_Two_tone_spec_DC_YOKO_0to8mA_Cav_7.45671GHz&-20dBm_QuBit3.5to6GHz&15dBm'
# path = directory + '\\' + measurement
# #
# # #Read data
# current = np.genfromtxt(path + '_CURRENT.csv')#*1e3 b
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
# #
# X,Y = np.meshgrid(current,freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
###############################################################################
clicked_data = np.array([
[0.053595, 5.777143],
[0.578831, 5.777143],
[1.811526, 5.744573],
[3.119254, 5.614293],
[4.587769, 5.451442],
[5.595363, 5.256021],
[5.959812, 5.174595],
[7.664147, 4.213775],
[7.867809, 4.067210],
[10.483266, 1.722157],
[11.287198, 1.021899],
[11.469422, 0.891618],
[11.715961, 0.679912],
[11.823152, 0.614772],
[12.123286, 0.484491],
[12.348387, 0.533346],
[12.905780, 0.859048],
[13.077285, 1.005614],
[13.666835, 1.543021],
[14.138474, 1.966434],
[0.064315, 8.692174],
[0.728898, 8.643318],
[1.586425, 8.529323],
[2.572581, 8.285046],
[3.451546, 7.943060],
[4.459140, 7.373082],
[5.070128, 6.949670],
[5.552487, 6.607683],
[6.377856, 5.988850],
[7.288978, 5.304876],
[7.417608, 5.207165],
[0.075034, 9.294722],
[0.568112, 9.245866],
[1.339886, 9.115586],
[3.001344, 8.447897],
[3.397950, 8.268761],
[4.212601, 7.747639],
[5.027251, 7.161376],
[5.423858, 6.868244],
[5.938374, 6.509972],
[6.388575, 6.200556],
[6.806620, 5.972564],
[7.085316, 5.825999],
[7.814214, 5.500297],
[0.064315, 10.955800],
[0.407325, 10.906945],
[1.629301, 10.483533],
[2.111660, 10.255542],
[2.861996, 9.848415],
[3.237164, 9.669279],
[3.923185, 9.294722],
[4.416263, 9.148156],
[4.769993, 9.001590],
[5.552487, 8.724744],
[6.152755, 8.529323],
[6.742305, 8.301332],
[7.514079, 8.008200],
[7.803495, 7.829064],
[7.899966, 7.747639],
[8.146505, 7.405652],
[0.085753, 11.460638],
[0.514516, 11.314072],
[0.889684, 11.216362],
[5.263071, 9.294722],
[5.595363, 9.148156],
[5.884778, 9.034160],
[7.578394, 8.171051],
[7.771337, 8.024485],
[8.125067, 7.812779],
[9.325605, 7.291656],
[9.615020, 7.161376],
[9.850840, 6.965955],
[9.968750, 6.884529],
[0.117910, 11.737484],
[0.750336, 11.542063],
[1.136223, 11.411783],
[1.500672, 11.297787],
[2.175974, 11.118651],
[2.808401, 10.972085],
[3.280040, 10.858090],
[5.638239, 9.913555],
[6.206351, 9.538998],
[6.570800, 9.262152],
[6.817339, 9.034160],
[7.406888, 8.513038],
[7.621270, 8.252476],
[7.846371, 8.089625],
[8.125067, 7.910490],
[8.307292, 7.829064],
[9.250571, 7.324226],
[9.615020, 7.128805],
[9.765087, 7.047380],
[9.893716, 6.949670],
[10.000907, 6.998525],
[10.301042, 6.819389],
[10.686929, 6.623968],
[9.647177, 7.259086],
[9.293448, 7.682498],
[8.125067, 8.985305],
[7.792776, 9.425002],
[5.498891, 11.851480],
[6.067003, 11.346642],
[6.313542, 11.069796],
[6.474328, 10.939515],
[6.495766, 11.884050],
[6.667272, 11.721199],
[8.092910, 10.271827],
[8.457359, 10.027551],
[9.218414, 9.441287],
[9.593582, 9.164441],
[9.915155, 8.855024],
[10.376075, 8.431612],
[10.761962, 7.894204],
[11.233602, 7.356797],
[11.555175, 6.982240],
[11.887466, 6.640253],
[12.262634, 6.526257],
[12.573488, 6.656538],
[12.016095, 6.558827],
[12.852184, 6.982240],
[13.109442, 7.259086],
[13.516767, 7.796494],
[13.881216, 8.219906],
[12.691398, 6.281981],
[13.077285, 6.412262],
[13.613239, 6.672823],
[13.795464, 6.770534],
[13.977688, 6.884529],
[13.977688, 10.011265],
[13.548925, 9.701849],
[12.970094, 9.359862],
[12.498454, 9.213296],
[11.919624, 9.213296],
[11.340793, 9.408717],
[10.740524, 9.734419],
[10.397513, 9.962410],
[9.936593, 10.336967],
[9.936593, 11.916620],
[10.183132, 11.574634],
[10.429671, 11.248932],
[11.533737, 9.864700],
[11.780276, 9.587853],
[11.994657, 9.408717],
[12.391263, 9.408717],
[12.744993, 9.750704],
[13.109442, 10.157831],
[0.224208, 4.655540],
[0.768259, 4.638602],
[1.505138, 4.630133],
[3.164839, 4.541212],
[4.666145, 4.426885],
[5.726701, 4.299854],
[6.539334, 4.143183],
[7.172913, 3.973809],
[7.579229, 3.812904],
[7.827152, 3.702811],
[8.939718, 5.125740],
[9.272009, 4.979174],
[9.625739, 4.783753],
[9.893716, 4.458052],
[10.204570, 3.855503],
[10.418952, 3.497232],
[10.558300, 3.285525],
[10.751243, 2.878398],
[11.340793, 1.852438],
[11.458703, 1.689587],
[13.441734, 2.568982],
[13.666835, 2.927254],
[13.924093, 3.432091],
[14.224227, 3.904359],
[14.610114, 4.588332],
[15.103192, 4.979174],
[15.553394, 5.142025],
[15.328293, 5.939994],
[15.124630, 5.598007],
[14.803058, 5.174595],
[14.631552, 4.962889],
[14.138474, 4.702328],
[13.591801, 4.458052],
[12.991532, 4.148635],
[12.723555, 3.985784],
[11.994657, 3.855503],
[11.480141, 4.083495],
[10.665491, 4.506907],
[9.829402, 4.897749],
[9.464953, 5.337446],
[9.164819, 5.777143],
[9.068347, 5.972564],
[12.809308, 5.125740],
[13.034409, 5.353731],
[13.195195, 5.516582],
[13.613239, 5.988850],
[9.668616, 7.421937],
[9.893716, 7.259086],
[10.204570, 7.031095],
[10.676210, 6.737963],
[13.645397, 6.721678],
[13.806183, 6.770534],
[13.977688, 6.884529],
[13.924093, 8.480467],
[13.784745, 8.268761],
[13.656116, 8.105911],
[10.633333, 8.268761],
[9.004032, 10.157831],
[9.422077, 9.864700],
[9.904435, 9.538998],
[10.268884, 9.262152],
[11.276478, 6.428547],
[11.608770, 6.298266],
[12.530612, 6.233126],
[12.916499, 6.347121],
[13.709711, 6.623968],
[13.977688, 6.819389],
[11.555175, 8.366472],
[11.919624, 8.154766],
[12.401983, 8.089625],
[12.777151, 8.301332],
[13.088004, 8.480467],
[8.939718, 11.281502],
[10.022345, 10.271827],
[10.611895, 9.832130],
[11.426546, 9.359862],
[11.994657, 9.197011],
[13.613239, 9.734419]
])
plt.errorbar(clicked_data[:,0],clicked_data[:,1], fmt='o', mfc='none', mew=1, mec='royalblue')
###############################################################################
directory = 'C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_AugustusIII.txt"
path = directory + '\\' + fname
energies = np.genfromtxt(path)
level_num = 20
current = np.linspace(0, 15,len(energies[:,0]))
for idx in range(1,level_num):
    # print(len(energies[:, idx]))
    plt.plot(current, energies[:,idx] - energies[:,0], color='k', linestyle ='--', alpha = 0.5)

###############################################################################
directory = 'C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_spectrum_AugustusIII_jc=0.txt"
path = directory + '\\' + fname
energies = np.genfromtxt(path)
level_num = 20
current = np.linspace(0, 15,len(energies[:,0]))
for idx in range(1,level_num):
    # print(len(energies[:, idx]))
    plt.plot(current, energies[:,idx] - energies[:,0], color='r', linestyle ='-', alpha = 0.5)

plt.xticks(size = 18.0)
plt.yticks(size = 18.0)
plt.xlim([0,14])
plt.ylim([0,12])
plt.yticks([0,4,8,12])
plt.xticks([0,4,8,12])
plt.show()