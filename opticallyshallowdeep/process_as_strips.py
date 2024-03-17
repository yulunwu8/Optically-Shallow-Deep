
import os, gc, warnings, math
import numpy as np
import pandas as pd
import netCDF4 as nc4
import warnings

from joblib import Parallel, delayed
from datetime import datetime

import scipy
from scipy import ndimage
from scipy.ndimage import uniform_filter
from scipy.ndimage import binary_dilation 

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model,load_model

def process_as_strips (full_img, image_path, if_SR, model_path, selected_columns, model_columns, file_in):
    strip1, strip2, strip3, strip4,strip5=make_vertical_strips(full_img) #create strips with overlap
    striplist=[strip1, strip2, strip3, strip4,strip5] #make strip list
    RGBlist=[]
    for n in range(len(striplist)):
        print(" Strip {}".format(n+1))
        strip_p=process_img_to_rgb(striplist[n],image_path, if_SR, model_path, selected_columns, model_columns, file_in) #output is RGB of image
        RGBlist.append(strip_p) #append processed strip to RGB list
    RGB_img=join_vertical_strips(RGBlist[0], RGBlist[1], RGBlist[2], RGBlist[3],RGBlist[4])
    plot_RGB_img(RGB_img) #show the final image
    return RGB_img


def make_vertical_strips(full_img):
    '''use to save ram, process bigger images faster, and it overlaps so middle image is not 
    distorted from how edge pixels are handled'''
    height, width, _ = full_img.shape #this is done so strips do not have artifacts from kernals
    overlap_size = 16 #size of overlap, max tile size is 15, so there is a 1px buffer
    strip1 = full_img[:, :width//5 + overlap_size, :]#left overlap
    strip2 = full_img[:, width//5: 2*width//5+ overlap_size, :]# left half overlap
    strip3 = full_img[:, 2*width//5:3*width//5 + overlap_size, :]#left overlap
    strip4 = full_img[:, 3*width//5:4*width//5+ overlap_size, :]#left overlap
    strip5 = full_img[:, 4*width//5:width, :]# no overlap
    return strip1,strip2,strip3,strip4,strip5

def process_img_to_rgb(img,file_path, if_SR, model_path, selected_columns, model_columns, file_in):
    img,img_name,correction=correct_baseline(img,file_path, if_SR, file_in)#used on slices or whole images
    final_cord=get_water_pix_coord(img,correction, if_SR) #getting coordinates of water pixels
    if len(final_cord)==0:
        RGB_img=make_blank_img(img)
        return RGB_img
    else:
        # print("  {} {} Coordinates of non-glinty water pixels".format(time_tracker(start_time),len(final_cord)))
        print("  {} Coordinates of non-glinty water pixels".format(len(final_cord)))
        
        filter_image = process_image_with_filters(img, selected_columns) #creating a filter image to extract values from
        edge_nodata_list = select_edge_and_buffer_no_data_pixels (img,correction, if_SR) #selecting pixels for slow processing
        value_list,cord_list=create_df_for_processing_v2(selected_columns, img_name, img,filter_image, final_cord,edge_nodata_list,correction, model_columns)
        del filter_image,edge_nodata_list,final_cord #line above makes coordinate and value list for making an image
        gc.collect()
        cord_list, pred_results, con_1=load_model_and_predict_pixels(value_list,model_path,cord_list, if_SR)
        RGB_img=make_output_images_fast(cord_list, pred_results, con_1,img)#make RBG image
        # print("  {} Finished model predictions".format(time_tracker(start_time)))
        print("  Finished model predictions")
        
        del cord_list, pred_results, con_1,img
        gc.collect()
        return RGB_img #this image is 0: OSW/ODW, 1:pred/prob, 2: Mask


def correct_baseline(img,file_path, if_SR, file_in):
    
    from xml.dom import minidom
    
    # if ACOLITE input 
    if if_SR:
        
        # Open the NetCDF file
        with nc4.Dataset(file_in, "r") as nc:
            tile_code = nc.getncattr('tile_code')
            img_name = tile_code[1:]
            
        imgf = img
        correction = 0
        
    # if L1C input 
    else:
        xml_path = os.path.join(file_in,'MTD_MSIL1C.xml')
        xml = minidom.parse(xml_path)#look at xml for correction first
        tdom = xml.getElementsByTagName('RADIO_ADD_OFFSET')#if this tag exists it is after baseline 4
        
        
        tdom_URI = xml.getElementsByTagName('PRODUCT_URI')
        S2_URI = tdom_URI[0].firstChild.nodeValue
        img_name = S2_URI[39:44]
        
        
        # If no RADIO_ADD_OFFSET
        if len(tdom) == 0: 
            
            def Add_1000(img):
                return img + 1000 #function for parallization in this function
        
            chunk_size = len(img) // 4 #split into 4 for four cores, apply correction this way
            chunks = [img[i:i + chunk_size] for i in range(0, len(img), chunk_size)]
            imgf = np.concatenate(Parallel(n_jobs=4)(delayed(Add_1000)(chunk) for chunk in chunks), axis=0)
            correction = 1000
            
            '''Correction is a very important variable, since in some of the images we need to add 1000 in order to
            correct for baseline 4. In these instances, 0 becomes 1000. There are times where we need to mask out 0 pixels
            or avoid 0, so we use correction as a variable for pixels that are originally 0'''
            print('  Adjusted pixel value for before Baseline 4 processing')
            del chunks
        
        # If there is RADIO_ADD_OFFSET
        else:
            imgf = img
            correction = 0

    del img
    return imgf, img_name, correction
    
def get_water_pix_coord(img,correction, if_SR):
    #creates the mask of what is water by using Glint threshold, NDWI, NDSI...
    if if_SR == False:
        glint_t= 1500#this glint thresholds were used when training the model.
    else:
        glint_t= 500 #glint threshold as per ACOLITE
    glint_coordinates = np.where(img[:, :, 9] < glint_t)#same threshold as in model
    glr, glc = glint_coordinates
    glint_coordinates_list = list(zip(glr, glc))#where not glint
    del glr, glc,glint_coordinates  
    b3,b8,b11 = img[:, :, 2].astype(np.float32),img[:, :, 7].astype(np.float32),img[:, :, 9].astype(np.float32)
    NDWI = (b3 - b8) / (b3 + b8 +1e-8) #NDWI with avoiding div 0
    coordinates_NDWI = np.where(NDWI > 0)#where water (used to be 0)
    ndwir, ndwic = coordinates_NDWI
    coordinate_list_NDWI = list(zip(ndwir,ndwic)) 
    del b8,NDWI, coordinates_NDWI,ndwir, ndwic
    gc.collect()
    NDSI = (b3 - b11) / (b3 + b11 +1e-8) #NDSI with avoiding div 0
    coordinates_NDSI = np.where(NDSI < .42)#where not snow
    ndsir, ndsic = coordinates_NDSI
    coordinate_list_NDSI = list(zip(ndsir,ndsic))
    del b3,b11,NDSI,coordinates_NDSI,ndsir,ndsic
    gc.collect()
    
    # L1C
    if if_SR == False:
        ND_coordinates = np.column_stack(np.where(np.all((img > correction) & (img < 30000), axis=-1)))#where not no data (in any band)
        ND_coordinates_list = list(map(tuple, ND_coordinates))
        common_coordinates_set = set(glint_coordinates_list) & set(ND_coordinates_list)& set(coordinate_list_NDWI)& set(coordinate_list_NDSI)
        common_coordinates_list = list(common_coordinates_set)  # Convert set to list
        del ND_coordinates,ND_coordinates_list,common_coordinates_set,glint_coordinates_list
    
    # L2R
    else:
        ND_coordinates = np.column_stack(np.where(np.all((img > -30000) & (img < 30000), axis=-1)))#where not no data (in any band)
        ND_coordinates_list = list(map(tuple, ND_coordinates))
        Acolite_pix=np.column_stack(np.where(np.all((img <= 3000), axis=-1)))#threshold from ACOLITE
        Acolite_pix_list = list(map(tuple, Acolite_pix))
        common_coordinates_set = set(glint_coordinates_list)&set(Acolite_pix_list)&set(ND_coordinates_list)&set(coordinate_list_NDWI)
        common_coordinates_list = list(common_coordinates_set)
        del common_coordinates_set,Acolite_pix,Acolite_pix_list,ND_coordinates,ND_coordinates_list,glint_coordinates_list
    gc.collect()
    return common_coordinates_list


def make_blank_img(img):
    Y_b, X_b, b = img.shape #sometimes the image is all no data or the correction value, in this instance, we make a blank image
    RGB_img = np.zeros((Y_b, X_b, 3), dtype=np.uint8)
    # print('  {} Blank strip added. No valid water pixels'.format(time_tracker(start_time)))
    print('  {} Blank strip added. No valid water pixels')
    return RGB_img

def time_tracker(start_time):
    return "{}sec".format(round((datetime.now() - start_time).total_seconds(), 2))


def process_image_with_filters(img, selected_columns):
    height, width, bands = img.shape#the output of this is an image of the filters required for this model
    filter_list = [value for value in selected_columns if value not in [["lat"], ["long"], ["lat_abs"]]]
    output_bands = []
    for band, kernel_size, filter_type in filter_list:
        
        if filter_type is None:
            filtered_band = img[:, :, band].astype(np.uint16)#this means it is a single pixel
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                filtered_band = apply_filter(img[:, :, band].astype(np.float32), kernel_size, filter_type)
                filtered_band[filtered_band==-32768] = 32768
                filtered_band = filtered_band.astype(np.uint16)
            
        output_bands.append(filtered_band)#append to list of filters
        del filtered_band
        gc.collect()
    del img
    output_image = np.stack(output_bands, axis=-1)# Stack along the last dimension to make filter image
    del output_bands
    gc.collect()
    return output_image.astype(np.uint16)

def apply_filter(image_band, kernel_size, filter_type):
    if filter_type == 'sd':
        return std_dev_filter(image_band, kernel_size)
    elif filter_type == 'avg':
        return uniform_filter(image_band, size=kernel_size, mode='mirror', origin=0) #avg filtering

def std_dev_filter(image_band, kernel_size):
    mean = uniform_filter(image_band, size=kernel_size, mode='mirror', origin=0)#custom std filter that is fast
    mean_of_squares = uniform_filter(image_band.astype(np.int32)**2, size=kernel_size, mode='mirror', origin=0)
    std_deviation = np.sqrt(np.maximum(mean_of_squares - mean**2, 0)) #calculation of std
    del mean, mean_of_squares
    gc.collect()
    return std_deviation

def select_edge_and_buffer_no_data_pixels (img,correction, if_SR):
    b3 = img[:, :, 2].copy()#function selects the edge and no data and makes a buffer. these pixels are processed the slow method
    if if_SR == False:
        condition_mask = np.logical_or(b3 <=correction, b3 > 30000) #filter for no data pixels -- this is simple since more complex can cause errors
    else:
        condition_mask = np.logical_or(b3 <-30000, b3 > 30000) #filter for nodata pixels
    dilated_mask = binary_dilation(condition_mask, iterations=8)#mask within 8 pixels of bad pixel
    del condition_mask
    final_coordinates_condition = np.argwhere(dilated_mask) #coordinates of bad pixels
    del dilated_mask
    edge_mask = np.zeros_like(b3, dtype=bool)#Create an edge mask
    edge_mask[:8, :] = True #Top edge
    edge_mask[-8:, :] = True #Bottom edge
    edge_mask[:, :8] = True #Left edge
    edge_mask[:, -8:] = True #Right edge
    final_coordinates_edge = np.argwhere(edge_mask)# Get coordinates of the pixels in the edge mask
    del edge_mask,b3
    edge_nodata_list = np.unique(np.concatenate([final_coordinates_condition, final_coordinates_edge]), axis=0).tolist()
    edge_nodata_list = list(map(tuple, edge_nodata_list))#get list of unique bad pixels
    del final_coordinates_condition
    gc.collect()
    return edge_nodata_list

def create_df_for_processing_v2 (selected_columns, img_name, img,filter_image, final_cord,edge_nodata_list,correction, model_columns):
    start_time = datetime.now()
    value_list,cord_list=[],[]
    columns_to_add = [['lat_abs'], ['lat'], ['long']]
    edge_nodata_list = list(set(edge_nodata_list) & set(final_cord))
    final_cord_not_in_edge_nodata = list(set(final_cord) - set(edge_nodata_list))#edge pixels and near no data need special treatment
    processed_edge_nodata_list = Parallel(n_jobs=-1)(delayed(process_edge_nodata_pixel)(img_name,cord,img,correction, selected_columns, model_columns) for cord in edge_nodata_list)
    if len(processed_edge_nodata_list)>0:
        values, cords = zip(*processed_edge_nodata_list)
        value_list.extend(values)
        cord_list.extend(cords)
    del edge_nodata_list,processed_edge_nodata_list
    final_cord_not_in_edge_nodata_array=np.array(final_cord_not_in_edge_nodata)#these are middle pixels and we can extract filter img values
    y_values = final_cord_not_in_edge_nodata_array[:, 0]# Extract y, x coordinates directly using array indexing
    x_values = final_cord_not_in_edge_nodata_array[:, 1]
    pixel_values = filter_image[y_values, x_values, :].tolist()# Process pixels and other operations using NumPy vectorized operations
    loc_list, labelling = get_location_values(img_name)
    for column_name in columns_to_add:
        if column_name in selected_columns:
            value_to_insert = int(loc_list[1]) if column_name == ['lat'] else int(loc_list[0]) if column_name == ['long'] else int(loc_list[2])
            for sublist in pixel_values:
                sublist.insert(0, value_to_insert)# Insert the location value at position 1 in all lists within pixel_values
    del y_values,x_values
    gc.collect()
    return value_list+pixel_values,cord_list+final_cord_not_in_edge_nodata

def process_edge_nodata_pixel(img_name,cord,img,correction, selected_columns, model_columns):
    y, x = cord[0], cord[1]#this is a function for joblib parallelization
    values, l = get_values_for_pixel(selected_columns, img_name, img, y, x,correction, model_columns)
    return values, cord


def get_values_for_pixel(selected_columns, img_name, img, y, x,correction, model_columns):
    values, labels = [], []#iterate throught the column names
    for n in range(len(selected_columns)):
        if selected_columns[n][0] in ['long', 'lat', 'lat_abs']: 
            location_list, labelling = get_location_values(img_name)#address the location values
            position = labelling.index(selected_columns[n][0])#appending location information
            values.append(location_list[position])
            labels.append(labelling[position])
        else:
            band = img[:, :, int(selected_columns[n][0])]#get bands, wsize, and function from column name
            window_size, function = int(selected_columns[n][1]), selected_columns[n][2]
            mid_size = window_size / 2
            Lv, Sv = math.ceil(mid_size), math.floor(mid_size)#this section alters the window for edge pixels
            region = band[max(0, y - Sv):min(y + Lv, band.shape[0]), max(0, x - Sv):min(x + Lv, band.shape[1])]
            result=region
            zero_value_mask = np.any(img[max(0, y - Sv):min(y + Lv, img.shape[0]),
                              max(0, x - Sv):min(x + Lv, img.shape[1]), :] == correction, axis=-1) #use of correction to find if any pixels are originally 0
            if np.any((region <= correction) | (region > 30000) | zero_value_mask): 
                condition_mask = ((region <= correction) | (region > 30000) | zero_value_mask)
                mask = ~((region <-30000) | (region > 30000)| zero_value_mask)#mask out no data pixels
                result = region[mask]
            if len(result) == 0:
                values.append(0)
            elif function == 'avg': #process if function is avg
                values.append(int(np.average(result, weights=np.ones_like(result) / result.size))) #calculate average
            elif function == 'sd': 
                std_value = np.std(result)#process if function is sd
                if not np.isnan(std_value):  
                    values.append(int(std_value)) #calculate std
                else:
                    values.append(0)#Handle NaN case
            elif function == None:
                values.append(band[y,x])#process if function it is pixel value
            del result,region,band
            labels.append(model_columns[n])
            
    if labelling == False:
        labels = []#save space by returning empty labels
    return values, labels

def get_location_values(img_name): 
    latitude_bands = {'C': -76,'D': -68,'E': -60,'F': -52,'G': -44,'H': -36,'J': -28,'K': -20,'L': -12,'M': -4,'N': 4,'P': 12,'Q': 20,'R': 28,'S': 36,'T': 44,'U': 52,'V': 60,'W': 68,'X': 76,}
    long_str=str(get_mean_longitude(img_name[:2]))#gets location info from name string
    long, lat=get_mean_longitude(img_name[:2]), latitude_bands[img_name[2]]
    lat_abs=abs(lat) #calculating absolute latitude
    location_list=[long,lat,lat_abs]
    labelling=['long','lat','lat_abs']
    del long_str,long, lat,lat_abs
    return location_list,labelling

def get_mean_longitude(utm_zone):
    utm_zone=int(utm_zone)
    if 1 <= utm_zone <= 60: #Check if the input UTM zone is within the valid range (1-60)
        mean_longitude = 180 - (utm_zone - 0.5) * 6#calculate the mean longitude for the given UTM zone
        return int(mean_longitude)
    else:
        return None# Return None for invalid input

def load_model_and_predict_pixels(value_list, model_path, cord_list, if_SR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    loaded_model=load_tf_model(model_path, if_SR)
    chunk_size = len(value_list) // 4 #breaking it into chunks makes it take up less ram
    chunks_value = [value_list[i:i + chunk_size] for i in range(0, len(value_list), chunk_size)]
    chunks_cord = [cord_list[i:i + chunk_size] for i in range(0, len(cord_list), chunk_size)]
    result_chunks = []
    for chunk_value, chunk_cord in zip(chunks_value, chunks_cord):
        value_arr= (np.array(chunk_value)/10000).astype('float16')
        df = pd.DataFrame(value_arr)
        with tf.device('/cpu:0'):
            pred_proba = loaded_model.predict_on_batch(df)#predprob in chunk
        pred_proba_np = np.array(pred_proba)
        con_1 = (pred_proba_np * 100).astype(int)#get our confidence in the way we like it
        pred_results = [1 if i >= 50 else 0 for i in con_1]#getting prediction results from pred_prob
        result_chunks.append((chunk_cord, pred_results, con_1))#append chunk
        del pred_results,con_1,pred_proba_np,df,value_arr
    cord_list, pred_results, con_1 = zip(*result_chunks)

    cord_list = np.concatenate(cord_list)
    pred_results = np.concatenate(pred_results)
    con_1 = np.concatenate(con_1)

    del chunks_value, chunks_cord, loaded_model, chunk_size

    return cord_list.tolist(), pred_results.tolist(), con_1.tolist()


def load_tf_model(model_path, if_SR):
    if if_SR == False:
        model=d6_model(32,0.01)
        model.load_weights(model_path)
        return model
    else:
        model=d4_model(64,0.01)
        model.load_weights(model_path)
        return model

def d6_model(u1,LR): # the last tested. good for more data.
    INPUT=Input(shape=(20,))
    d1=Dense(u1, activation='LeakyReLU')(INPUT)
    d2=Dense(u1, activation='LeakyReLU')(d1)
    d3=Dense(u1, activation='LeakyReLU')(d2)
    d4=Dense(u1, activation='LeakyReLU')(d3)
    d5=Dense(u1, activation='LeakyReLU')(d4)
    d6=Dense(u1, activation='LeakyReLU')(d5)
    d7=Dense(1, activation='sigmoid')(d6)
    model=Model(inputs=[INPUT], outputs=[d7])
    return model

def d4_model(u1,LR): # the last tested. good for more data.
    INPUT=Input(shape=(13,))
    d1=Dense(u1, activation='LeakyReLU')(INPUT)
    d2=Dense(u1, activation='LeakyReLU')(d1)
    d3=Dense(u1, activation='LeakyReLU')(d2)
    d4=Dense(u1, activation='LeakyReLU')(d3)
    d5=Dense(1, activation='sigmoid')(d4)
    model=Model(inputs=[INPUT], outputs=[d5])
    return model

def make_output_images_fast(cord_list, preds, pred_probs, img):
    Y_b, X_b, b = img.shape
    pred_image = np.zeros((Y_b, X_b, 1), dtype=np.uint8)#makes blank images
    predprob_image = np.zeros((Y_b, X_b, 1), dtype=np.uint8)
    masked_image = np.zeros((Y_b, X_b, 1), dtype=np.uint8)
    cords = np.array(cord_list)
    pred_image[cords[:, 0], cords[:, 1], :] = np.array(preds).reshape(-1, 1) #applies the values to the coordinate
    predprob_image[cords[:, 0], cords[:, 1], :] = np.array(pred_probs).reshape(-1, 1)
    masked_image[cords[:, 0], cords[:, 1], :] = 1 #sets all pixels that were used for OSW/ODW ==1
    RGB_img = np.concatenate([pred_image, predprob_image, masked_image], axis=-1) #make RGB image
    del pred_image,predprob_image,masked_image,cords
    gc.collect()
    return RGB_img

def join_vertical_strips(strip1, strip2, strip3, strip4, strip5):
    overlap_size = 16 #this is done so strips do not have artifacts from kernals
    strip1_cropped = strip1[:, :strip1.shape[1] -overlap_size//2, :]#crop 8 from left
    strip2_cropped = strip2[:, overlap_size//2: -overlap_size//2, :]#crop 8 from left and 8 right
    strip3_cropped = strip3[:, overlap_size//2: -overlap_size//2, :]#crop 8 left and 8 right
    strip4_cropped = strip4[:, overlap_size//2: -overlap_size//2, :]#crop 8 left and 8 right   
    strip5_cropped = strip5[:, overlap_size//2:, :] #crop 8 right, subtracted 48pix
    joined_array = np.hstack((strip1_cropped, strip2_cropped, strip3_cropped, strip4_cropped,strip5_cropped))
    return joined_array

def plot_RGB_img(RGB_img):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
    ax[0].imshow(RGB_img[:,:,0])#plotting to see what OSW/ODW looks like
    ax[0].set_title('Prediction Image')
    ax[1].imshow(RGB_img[:,:,1])
    ax[1].set_title('Prediction Probability Image')
    ax[2].imshow(RGB_img[:,:,2])
    ax[2].set_title('Masked Image')
    plt.show()



