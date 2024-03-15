

import sys, os, gc
import tifffile as tif


from .make_multiband_image import make_multiband_image
from .check_transpose import check_transpose

from .process_as_strips import process_as_strips
from .parse_string import parse_string

from .write_georef_image import write_georef_image


def run(file_in,folder_out):
    
    print('test')
    
    
    ### Add logging 
    
    
    
    
    ### Check the two 
    # folder_out: if not exist -> create it 
    
    
    
    
    
    
    
    # Columns 
    GTOA_model_columns=['long', 'lat_abs', 'B2-w_15_sd', 'B3-w_3_sd', 'B3-w_7_avg', 'B3-w_9_avg', 'B3-w_11_sd', 'B3-w_15_sd', 'B4-w_5_avg', 'B4-w_11_sd', 'B4-w_13_avg', 'B4-w_13_sd', 'B4-w_15_sd', 'B5-w_13_sd', 'B5-w_15_sd', 'B8-w_9_sd', 'B8-w_13_sd', 'B8-w_15_sd', 'B11-w_9_avg', 'B11-w_15_sd']
    GACOLITE_model_columns= ['lat_abs', 'B1-w_15_sd', 'B2-w_1', 'B3-w_5_sd', 'B3-w_7_sd', 'B3-w_13_sd', 'B3-w_15_sd', 'B4-w_15_sd', 'B5-w_11_avg', 'B5-w_15_avg', 'B5-w_15_sd', 'B8-w_13_sd', 'B11-w_15_avg']       
    
    
    
    
    
    
    # take input path 
    
    # identify model path 
    
    if file_in.endswith('.safe') or file_in.endswith('.SAFE'):
        if 'MSIL1C' in file_in: 
            if_SR = False
            model = 'models/TOA.h5'
            model_columns = GTOA_model_columns
            
            
    ### else: SR ???
    else:
        if_SR = True
        model = 'models/SR.h5'
        model_columns = GACOLITE_model_columns
        
        
    # Make it a list of lists
    selected_columns = [parse_string(s) for s in model_columns]
    
    
    
    print('if_SR: ' +str(if_SR))
    
    print('model: ' +str(model))
    
    model_path = os.path.join(os.path.dirname(__file__), model)    
    print('model_path: ' +str(model_path))
    
    
    # copy files to a new path??? Why 
    
    # make multiband_image (should include ACOLITE NetCDF as input) !!! 
    image_path = make_multiband_image(file_in,folder_out)
    
    print('\nmultiband_image: '+str(image_path))
    
    
    #read image
    image = tif.imread(image_path, dtype='int16') 
    
    # check 
    image=check_transpose(image)

    #create strips and process them -- make big RGB image
    RGB_img=process_as_strips(image, image_path, if_SR, model_path, selected_columns, model_columns) 
    
    

    write_georef_image(image_path,RGB_img,image_path[:-4]+'OSW_ODW.tif') #write as geotiff
    
    print("Image OSW/ODW completed {}".format(RGB_img.shape))
    
    del RGB_img
    gc.collect()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




