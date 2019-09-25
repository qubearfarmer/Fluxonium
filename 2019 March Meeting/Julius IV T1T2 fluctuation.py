import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=[7,2.5])
t1_array = []
t1_error_array = []
t2_array = []
t2_error_array = []
#########################################################################
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\pre_heroT1.txt'
# t1_array = np.append(t1_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\pre_heroT1err.txt'
# t1_error_array = np.append(t1_error_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\pre_heroT2.txt'
# t2_array = np.append(t2_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\pre_heroT2err.txt'
# t2_error_array = np.append(t2_error_array,np.genfromtxt(path))
#########################################################################
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT1_0.txt'
t1_array = np.append(t1_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT1err_0.txt'
t1_error_array = np.append(t1_error_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT2_0.txt'
t2_array = np.append(t2_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT2err_0.txt'
t2_error_array = np.append(t2_error_array,np.genfromtxt(path))
#########################################################################
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT1_1.txt'
t1_array = np.append(t1_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT1err_1.txt'
t1_error_array = np.append(t1_error_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT2_1.txt'
t2_array = np.append(t2_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT2err_1.txt'
t2_error_array = np.append(t2_error_array,np.genfromtxt(path))
#########################################################################
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT1_2.txt'
t1_array = np.append(t1_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT1err_2.txt'
t1_error_array = np.append(t1_error_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT2_2.txt'
t2_array = np.append(t2_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT2err_2.txt'
t2_error_array = np.append(t2_error_array,np.genfromtxt(path))
#########################################################################
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT1_3.txt'
t1_array = np.append(t1_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT1err_3.txt'
t1_error_array = np.append(t1_error_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT2_3.txt'
t2_array = np.append(t2_array,np.genfromtxt(path))
path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\heroT2err_3.txt'
t2_error_array = np.append(t2_error_array,np.genfromtxt(path))
#########################################################################
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\post_heroT1_1.txt'
# t1_array = np.append(t1_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\post_heroT1err_1.txt'
# t1_error_array = np.append(t1_error_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\post_heroT2_1.txt'
# t2_array = np.append(t2_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\post_heroT2err_1.txt'
# t2_error_array = np.append(t2_error_array,np.genfromtxt(path))
#########################################################################
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\post_heroT1_2.txt'
# t1_array = np.append(t1_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\post_heroT1err_2.txt'
# t1_error_array = np.append(t1_error_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\post_heroT2_2.txt'
# t2_array = np.append(t2_array,np.genfromtxt(path))
# path = 'G:\Projects\Fluxonium\Data\Julius IV\Summary\post_heroT2err_2.txt'
# t2_error_array = np.append(t2_error_array,np.genfromtxt(path))

loop_index = np.linspace(0,len(t1_array),len(t1_array))
plt.errorbar(loop_index, t1_array, yerr = t1_error_array,fmt = 's', mfc = 'none', mew = 2.0, mec = 'b', ecolor = 'b', label = r'$T_1$')
plt.errorbar(loop_index, t2_array, yerr = t2_error_array,fmt = 'd', mfc = 'none', mew = 2.0, mec = 'g', ecolor = 'g', label = r'$T_2e$')
# plt.tick_params(labelsize = 16.0)
# plt.ylim([0,800])
# plt.yticks([0,200,400,600,800])
plt.tick_params(labelsize = 16)
plt.ylim([0,800])
plt.yticks([0,200,400,600,800])
plt.xticks([0,10,20,30])
plt.xlim([-0.5,30.5])
# plt.legend()

plt.show()