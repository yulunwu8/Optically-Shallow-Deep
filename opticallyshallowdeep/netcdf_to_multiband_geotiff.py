


# def ACOLITE_to_Geotiff(file_in):
    
    
#     print('test')
    
import os
import numpy as np
import rasterio

from rasterio.crs import CRS
from rasterio.transform import from_origin
import netCDF4 as nc4

# from pyproj import Proj, transform
from pyproj import Proj, transform

from .find_epsg import find_epsg


def netcdf_to_multiband_geotiff(netcdf_file, folder_out):
    
    tif_base = os.path.basename(netcdf_file).replace('.nc','.tif')
    
    output_geotiff_file = os.path.join(folder_out, tif_base)
    
    
    value_for_nodata = -32768
    
    # Open the NetCDF file
    with nc4.Dataset(netcdf_file, "r") as nc:
        
        # test = nc.variables.keys()
        
        wkt = nc.variables['transverse_mercator'].getncattr('crs_wkt')
        
        
        sensor = nc.getncattr('sensor')
        global_dims = nc.getncattr('global_dims')
        height, width = global_dims.astype(int)
        
        # Sensor-specific band configuration
        if sensor in ['S2A_MSI', 'S2B_MSI']:
            bands = [443,492,560,665,704,740,783,833,865,1614,2202] if sensor == 'S2A_MSI' else [442,492,559,665,704,739,780,833,864,1610,2186]
        else:
            raise ValueError("Unsupported sensor")
        
        band_names = ['rhos_' + str(band) for band in bands]
        data_array = np.ma.empty((len(bands), height, width))
        
        for i, band_name in enumerate(band_names):
            
            
            ar = nc.variables[band_name][:,:] * 10_000
            
            # replace no data 
            
            ar[np.isnan(ar)] = value_for_nodata
            
            
            data_array[i] = ar
        
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
        
        
        
        
    ### Automate this!!! 
    # epsg_code = 32615
    
    
    epsg_code = find_epsg(wkt)
    
    
    
    # Initialize the projections
    proj_latlon = Proj(proj='latlong', datum='WGS84')
    
    
    # proj_utm = Proj(init='epsg:32615')
    
    proj_utm = Proj('epsg:' + str(epsg_code))
    
    
    # Transform the latitude and longitude to the target projection (EPSG:32615)
    # For this, find the min/max coordinates in the projected system
    xmin, ymin = transform(proj_latlon, proj_utm, lon[10979,0], lat[10979,0])
    xmax, ymax = transform(proj_latlon, proj_utm, lon[0,10979], lat[0,10979])
    
    # transform_ = from_origin(xmin-5, ymax+5, 10, 10)
    
    transform_ = from_origin(round(xmin-5,-1), round(ymax+5,-1), 10, 10)
    
    
    # transform = from_origin(lon.min(), lat.max(), lon.ptp() / lon.shape[1], lat.ptp() / lat.shape[0])


    
    with rasterio.open(
        output_geotiff_file, 
        'w', 
        driver='GTiff', 
        height=height, 
        width=width, 
        count=len(bands),
        
        # compress="lzw",
        
        # dtype=str(data_array.dtype), 
        dtype=rasterio.int16,
        nodata = value_for_nodata, 
        
        
        # crs='+proj=latlong', 
        
        crs = CRS.from_epsg(epsg_code),
        transform=transform_
    ) as dst:
        for i in range(len(bands)):
            dst.write(data_array[i,:,:], i+1)
            
    print(f'Multi-band GeoTIFF saved to {output_geotiff_file}')
    
    return output_geotiff_file
    

# # Example usage
# netcdf_file = 'path_to_your_netcdf_file.nc'
# output_geotiff_file = 'output_geotiff_file.tif'
# netcdf_to_multiband_geotiff(netcdf_file, output_geotiff_file)







