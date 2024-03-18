
import rasterio, gc
import numpy as np

def write_georef_image(image_path,RGB_img):
    
    output_name = image_path.replace('.tif','_OSW_ODW.tif')
    raster_with_ref = rasterio.open(image_path) # Open the raster with geospatial information
    crs = raster_with_ref.crs#Get the CRS (Coordinate Reference System) from the raster
    epsg_from_raster = crs.to_epsg()#Use the EPSG code from the CRS
    height, width, _ = RGB_img.shape
    count = 3 #3 bands for all the 
    dtype = RGB_img.dtype
    transform = raster_with_ref.transform# Use the same transform as the reference raster
    with rasterio.open(output_name, "w",driver="GTiff",height=height,width=width,
                       count=count,dtype=dtype,crs=crs,transform=transform) as dst:
        dst.write(np.moveaxis(RGB_img, -1, 0))
    del raster_with_ref
    gc.collect()
    
    