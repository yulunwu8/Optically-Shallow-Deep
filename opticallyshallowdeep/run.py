

import sys, os



def run(file_in,file_out):
    
    print('test')
    
    
    # take input path 
    
    # identify model path 
    
    if file_in.endswith('.safe') or file_in.endswith('.SAFE'):
        if 'MSIL1C' in file_in: 
            file_type = 'TOA'
            model = 'models/TOA.h5'
            
    ### else: SR ???
    else:
        file_type = 'SR'
        model = 'models/SR.h5'
    
    print('file_type: ' +str(file_type))
    
    print('model: ' +str(model))
    
    model_path = os.path.join(os.path.dirname(__file__), model)    
    print('model_path: ' +str(model_path))
    
    
    # copy files to a new path??? Why 
    
    
    # make multiband_image (should include ACOLITE NetCDF as input) !!! 
    
    
    
    
    
    
    # select model columns 
    
    
    
    
    
    
    
    
    
    




