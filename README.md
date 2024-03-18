# Optically-Shallow-Deep 

This python tool delineates optically shallow and deep waters in Sentinel-2 imagery. The tool uses a deep neural network that was trained on a diverse set of global images.

Supported input includes L1C SAFE files and ACOLITE-processed L2R netCDF files. The output geotiff contains probabilities of water pixels being optically shallow and deep. 

Originally coded by by Galen Richardson and Anders Knudby, modified and packaged by Yulun Wu

Home page: <a href="https://github.com/yulunwu8/Optically-Shallow-Deep" target="_blank">https://github.com/yulunwu8/Optically-Shallow-Deep</a>


 
## Installation 

**1 - Create a conda environment and activate it:**

```bash
conda create --name opticallyshallowdeep python=3.10
conda activate opticallyshallowdeep
```

**2 - Install tensorflow**

For mac OS: 

```bash
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos

```


For windows:

```bash
pip3 install tensorflow==2.13.0

```

In case of compatibility issues, please try the newest version of tensorflow: 

```bash
pip3 install --upgrade --force-reinstall tensorflow

```


For Linux and more on installing tensorflow: [https://www.tensorflow.org/install](https://www.tensorflow.org/install)


**3 - Install opticallyshallowdeep:**

```bash
pip3 install opticallyshallowdeep
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

A log file, an intermediate multi-band geotiff, and a preview PNG are also generated in the output folder. They can be deleted after the processing. 
