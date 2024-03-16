# Go up by 2 directory and import 

import sys
import os.path as path
two_up =  path.abspath(path.join(__file__ ,"../.."))
sys.path.append(two_up)

import opticallyshallowdeep as osd





file_in = '/Users/yw/Local_storage/temp_OSD_test/S2B_MSIL1C_20210902T015619_N0301_R117_T51KWB_20210902T033620.SAFE'
file_in = '/Users/yw/Local_storage/temp_OSD_test/temp_out/S2B_MSI_2021_09_02_02_02_11_T51KWB_L2R.nc'



folder_out = '/Users/yw/Local_storage/temp_OSD_test/temp_out2'


osd.run(file_in,folder_out)





'''

from matplotlib import pyplot as plt
plt.imshow(filtered_band, interpolation='nearest')
plt.show()



'''



