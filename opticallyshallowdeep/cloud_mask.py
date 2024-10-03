import os, sys
import rasterio
import numpy as np
from scipy import ndimage

def cloud_mask(file_L1C, buffer_size = 8):
    
    print('Making cloud mask...')
    
    files = os.listdir(file_L1C)
    metadata = {}
    metadata['file_L1C'] = file_L1C
    
    # Identify paths 
    for i, fname in enumerate(files):
        tmp = fname.split('.')
        path = '{}/{}'.format(file_L1C,fname)
        
        # Granules
        if (fname == 'GRANULE'):
            granules = os.listdir(path)
            
            # Check if there is only one granule file 
            n_granule = 0
            
            for granule in granules:
                if granule[0]=='.':continue
                
                n_granule += 1
                if n_granule>1: sys.exit('Warning: more than 1 granule')
                
                metadata['granule'] = '{}/{}/{}/IMG_DATA/'.format(file_L1C,fname,granule)
                metadata['MGRS_tile'] = granule.split('_')[1][1:]
                metadata['QI_DATA'] = '{}/{}/{}/QI_DATA'.format(file_L1C,fname,granule)
                
                
                # # MGRS
                # tile = metadata['MGRS_tile'] + '54905490'
                # d = m.toLatLon(tile)
                # metadata['lat'] = d[0]
                # metadata['lon'] = d[1]
                
                # Band files 
                image_files = os.listdir(metadata['granule'])
                for image in image_files: 
                    if image[0]=='.':continue
                    if image[-4:]=='.xml':continue
                    tmp = image.split('_')
                    metadata[tmp[-1][0:3]] = '{}/{}/{}/IMG_DATA/{}'.format(file_L1C,fname,granule,image)
    
    ### Load built-in mask
    
    gml_file = "{}/MSK_CLOUDS_B00.gml".format(metadata['QI_DATA'])
    jp2_file = "{}/MSK_CLASSI_B00.jp2".format(metadata['QI_DATA'])
    
    # For imagery before processing baseline 4: Jan 25, 2022
    if os.path.exists(gml_file): 

        # Built-in cloud mask 
        import geopandas as gpd
        from rasterio.features import geometry_mask
        
        # Load a raster as the base of the mask 
        image = metadata['B02']
        
        with rasterio.open(image) as src:
            # Read the raster data and transform
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs
        
        try: 
            # Read GML file 
            gdf = gpd.read_file(gml_file)
    
            # Create a mask using the GML polygons and the GeoTIFF metadata
            mask_cloud = geometry_mask(gdf['geometry'], transform=transform, out_shape=raster_data.shape, invert=True)
        
        # Sometimes the GML file contains no information, assume no clouds in such case
        except:
            mask_cloud = np.zeros_like(raster_data)
        
    # For imagery processing baseline 4
    elif os.path.exists(jp2_file): 
        band_ds = rasterio.open(jp2_file)
        band_array = band_ds.read(1)
        mask_cloud = band_array == 1
        mask_cloud = np.repeat(np.repeat(mask_cloud, 6, axis=0), 6, axis=1)
        
    else:
        sys.exit('Warning: cloud mask missing in {}.'.format(metadata['QI_DATA']))

    # To buffer: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html
    struct1 = ndimage.generate_binary_structure(2, 1)
    mask_cloud_buffered = ndimage.binary_dilation(mask_cloud, structure=struct1,iterations=buffer_size).astype(mask_cloud.dtype)
    
    print('Done')
    return mask_cloud_buffered