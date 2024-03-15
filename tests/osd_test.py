# Go up by 2 directory and import 

import sys
import os.path as path
two_up =  path.abspath(path.join(__file__ ,"../.."))
sys.path.append(two_up)

import opticallyshallowdeep as osd





file_in = '/Users/yw/Local_storage/temp_OSD_test/S2B_MSIL1C_20210902T015619_N0301_R117_T51KWB_20210902T033620.SAFE'
folder_out = '/Users/yw/Local_storage/temp_OSD_test/temp_out'


osd.run(file_in,folder_out)







