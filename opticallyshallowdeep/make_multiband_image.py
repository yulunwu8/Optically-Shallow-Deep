import numpy as np
import os
import glob
import rasterio
import gc

# make multiband images in output dir
def make_multiband_image(file_in,folder_out):

    basename = os.path.basename(file_in).rstrip(".SAFE")
    
    # output path
    imageFile = os.path.join(folder_out,basename) + '.tif'
    
    if os.path.exists(imageFile):
        print('Multi-band geotiff exists: ' + str(imageFile))
    else:
        print('Making multi-band geotiff: ' + str(imageFile))

        band_numbers = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        S2Files = [glob.glob(f'{file_in}/**/IMG_DATA/**/*{band}.jp2', recursive=True)[0] for band in band_numbers]
        b2File = S2Files[1]
        # Get image info
        band2 = rasterio.open(b2File)  # b2 has 10m spatial resolution
        crs = band2.crs
        res = int(band2.transform[0])
        arrayList = []
        for bandFile in S2Files:
            # print("Reading band: {}".format(bandFile))
            band = rasterio.open(bandFile)
            ar = band.read(1)
            bandRes = int(band.transform[0])
            if bandRes == res:
                ar = ar.astype('int16')
                arrayList.append(ar)
            elif bandRes > res:
                finerRatio = int(bandRes / res)
                ar = np.kron(ar, np.ones((finerRatio, finerRatio), dtype='int16')).astype('int16')
                arrayList.append(ar)
            del ar
            band.close()
        stack = np.dstack(arrayList)# To write to file, rasterio expects (bands, rows, cols), while dstack creates (rows, cols, bands)
        stackTransposed = stack.transpose(2, 0, 1)
        with rasterio.Env():
            profile = band2.profile #use band2 as an example
            profile.update(driver="GTiff",
                           count=len(S2Files),
                           compress="lzw")
            with rasterio.open(imageFile, 'w', **profile) as dst:
                dst.write(stackTransposed)
        band2.close()
        del stack,stackTransposed,S2Files,band2
        gc.collect()
        print('Done')

    return imageFile


