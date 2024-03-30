# Go up by 2 directory and import 

import sys
import os.path as path
two_up =  path.abspath(path.join(__file__ ,"../.."))
sys.path.append(two_up)

import opticallyshallowdeep as osd



file_L1C = '/Users/yw/Local_storage/temp_OSD_test/S2A_MSIL1C_20181019T051821_N0206_R062_T43PDQ_20181019T082135.SAFE'
file_L2R = '/Users/yw/Local_storage/temp_OSD_test/ACed/20181019/S2A_MSI_2018_10_19_05_26_21_T43PDQ_L2R.nc'


# file_L1C = '/Users/yw/Local_storage/temp_OSD_test/S2B_MSIL1C_20230928T153619_N0509_R068_T17MNP_20230928T205247.SAFE'
# file_L2R = '/Users/yw/Local_storage/temp_OSD_test/ACed/20230928/S2B_MSI_2023_09_28_15_44_58_T17MNP_L2R.nc'

folder_out = '/Users/yw/Local_storage/temp_OSD_test/temp_out3'

osd.run(file_L1C, folder_out, file_L2R=file_L2R)








'''


# Interactive mode
%matplotlib qt

# Inline plotting 
%matplotlib inline





from matplotlib import pyplot as plt
plt.imshow(img[:,:,0], interpolation='nearest')
plt.show()





# Windows gap problem 

n_strip = 2

test = striplist[n_strip]
test1 = test[:,:,9]

np.mean(test1)

from matplotlib import pyplot as plt
plt.imshow(test1, interpolation='nearest')
plt.show()


# cloud 
test2 = cloud_list[n_strip]
plt.imshow(test2, interpolation='nearest')
plt.show()




plt.imshow(strip_p[:,:,1], interpolation='nearest')
plt.show()

'''



