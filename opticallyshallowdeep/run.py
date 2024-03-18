
import sys, os, gc, time
import tifffile as tif
from importlib.metadata import version

from .make_multiband_image import make_multiband_image
from .check_transpose import check_transpose

from .process_as_strips import process_as_strips
from .parse_string import parse_string

from .write_georef_image import write_georef_image
from .netcdf_to_multiband_geotiff import netcdf_to_multiband_geotiff


def run(file_in,folder_out, to_log=True):
    
    ### Check the two 
    if not os.path.exists(file_in):
        sys.exit('file_in does not exist: ' + str(file_in))
    
    # folder_out: if not exist -> create it 
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    
    ### Print all metadata/settings and save them in a txt file
    if to_log: 
        # Start logging in txt file
        orig_stdout = sys.stdout
        
        log_base = os.path.basename(file_in).replace('.nc','.txt').replace('.safe','.txt').replace('.SAFE','.txt')
        log_base = 'OSD_log_'+ log_base
        log_file = os.path.join(folder_out,log_base)

        class Logger:
            def __init__(self, filename):
                self.console = sys.stdout
                self.file = open(filename, 'w')
                self.file.flush()
            def write(self, message):
                self.console.write(message)
                self.file.write(message)
            def flush(self):
                self.console.flush()
                self.file.flush()
    
        sys.stdout = Logger(log_file)
    
    # Metadata
    print('\n=== ENVIRONMENT ===')
    print('OSD version: ' + str(version('opticallyshallowdeep')))
    print('Start time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    print('file_in: ' + str(file_in))
    print('folder_out: ' + str(folder_out))
    print('\n=== PRE-PROCESSING ===')
    
    
    # Columns 
    GTOA_model_columns=['long', 'lat_abs', 'B2-w_15_sd', 'B3-w_3_sd', 'B3-w_7_avg', 'B3-w_9_avg', 'B3-w_11_sd', 'B3-w_15_sd', 'B4-w_5_avg', 'B4-w_11_sd', 'B4-w_13_avg', 'B4-w_13_sd', 'B4-w_15_sd', 'B5-w_13_sd', 'B5-w_15_sd', 'B8-w_9_sd', 'B8-w_13_sd', 'B8-w_15_sd', 'B11-w_9_avg', 'B11-w_15_sd']
    GACOLITE_model_columns= ['lat_abs', 'B1-w_15_sd', 'B2-w_1', 'B3-w_5_sd', 'B3-w_7_sd', 'B3-w_13_sd', 'B3-w_15_sd', 'B4-w_15_sd', 'B5-w_11_avg', 'B5-w_15_avg', 'B5-w_15_sd', 'B8-w_13_sd', 'B11-w_15_avg']       
    
    ### Take input path and identify model path 
    
    # TOA
    if (file_in.endswith('.safe') or file_in.endswith('.SAFE')) and 'MSIL1C' in file_in: 
        if_SR = False
        model = 'models/TOA.h5'
        model_columns = GTOA_model_columns
        
        # make multiband_image 
        image_path = make_multiband_image(file_in,folder_out)
        
    # SR 
    elif file_in.endswith('.nc') or file_in.endswith('.NC'):
        if_SR = True
        model = 'models/SR.h5'
        model_columns = GACOLITE_model_columns
        
        # make multiband_image 
        image_path = netcdf_to_multiband_geotiff(file_in, folder_out)
         
    # make it a list of lists
    selected_columns = [parse_string(s) for s in model_columns]
    print('If input is SR product: ' +str(if_SR))
    model_path = os.path.join(os.path.dirname(__file__), model)    
    print('Trained model to use: ' +str(model_path))
    
    # read image
    image = tif.imread(image_path, dtype='int16') 
    
    # check 
    image=check_transpose(image)

    print('\n=== PREDICTING OSW/ODW ===')

    # create strips and process them -- make big RGB image
    RGB_img=process_as_strips(image, image_path, if_SR, model_path, selected_columns, model_columns, file_in) 
    
    # write as geotiff
    write_georef_image(image_path,RGB_img) 
    print("Image OSW/ODW completed, dimension: {}".format(RGB_img.shape))
    
    print('Finish time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    
    del RGB_img
    gc.collect()
    
    # stop logging 
    if to_log: sys.stdout = orig_stdout
    
