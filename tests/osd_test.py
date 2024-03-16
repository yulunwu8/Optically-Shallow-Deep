# Go up by 2 directory and import 

import sys
import os.path as path
two_up =  path.abspath(path.join(__file__ ,"../.."))
sys.path.append(two_up)

import opticallyshallowdeep as osd





# file_in = '/Users/yw/Local_storage/temp_OSD_test/S2B_MSIL1C_20210902T015619_N0301_R117_T51KWB_20210902T033620.SAFE'

# file_in = '/Users/yw/Local_storage/221220_ACIX/230329 AE_correction/230402 Full_comparison/ACOLITE_only/S2A_MSI_2020_09_21_17_20_57_T15TVM_L2R.nc'
file_in = '/Users/yw/Local_storage/Tmart_paper/Lakes_data/ACOLITE_processing/Image_U/S2A_MSI_2020_09_21_17_21_01_T15TUM_L2R.nc'



folder_out = '/Users/yw/Local_storage/temp_OSD_test/temp_out'


osd.run(file_in,folder_out)







'''

from matplotlib import pyplot as plt
plt.imshow(data_array[5], interpolation='nearest')
plt.show()



'''



