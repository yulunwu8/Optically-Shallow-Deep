# Go up by 2 directories and import 

import sys
import os.path as path
two_up =  path.abspath(path.join(__file__ ,"../.."))
sys.path.append(two_up)

import opticallyshallowdeep as osd




file_L1C = '/Users/yw/Local_storage/temp_OSD_test/S2B_MSIL1C_20230928T153619_N0509_R068_T17MNP_20230928T205247.SAFE'
file_L2R = '/Users/yw/Local_storage/temp_OSD_test/ACed/20230928/S2B_MSI_2023_09_28_15_44_58_T17MNP_L2R.nc'

folder_out = '/Users/yw/Local_storage/temp_OSD_test/temp_out5'

osd.run(file_L1C, folder_out, file_L2R=file_L2R)













