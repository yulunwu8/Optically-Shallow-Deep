# Optically-Shallow-Deep 

This python tool delineates optically shallow and deep waters in Sentinel-2 imagery. The tool uses a deep neural network that was trained on a diverse set of global images.

Supported input includes L1C SAFE files and ACOLITE-processed L2R netCDF files. The output geotiff contains probabilities of water pixels being optically shallow and deep. 

Originally coded by by Galen Richardson and Anders Knudby, modified and packaged by Yulun Wu


 
## Installation 

**1 - Create a conda environment and activate it:**

```bash
conda create --name opticallyshallowdeep
conda activate opticallyshallowdeep
```

**2 - Install tensorflow**

For mac OS: 

```bash
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos

```

(Optional) To utilize GPUs on mac OS: 


```bash
python -m pip install tensorflow-metal

```



For windows:

```bash
python3 -m pip install tensorflow

```



More on installing tensorflow: [https://www.tensorflow.org/install](https://www.tensorflow.org/install)


**3 - Install other dependencies:**

```bash
conda install -c conda-forge geopandas rasterio tifffile netCDF4 pyproj
```


**4 - Install opticallyshallowdeep:**

```bash
python3 -m pip install opticallyshallowdeep
```


## Quick Start

```python
import opticallyshallowdeep as osd

# Input file 
file_in = 'test_folder_in/S2.SAFE' # or path to an ACOLTIE-generated L2R netCDF file

# Output folder 
folder_out = 'folder/test_folder_out'

# Run the OSW/ODW classifier 
osd.run(file_in, folder_out)
```


Output is a 3-band geotiff: 

- B1: Binary prediction (OSW/ODW)
- B2: Prediction probability of OSW (100 means most likely OSW, 0 means most likely ODW) 
- B3: pixels that are masked out

An intermediate multi-band geotiff and a log file are also generated in the output folder. They can be deleted after the processing. 
