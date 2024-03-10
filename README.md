# Optically-Shallow-Deep 

This python tool delineates optically shallow and deep waters in Sentinel-2 imagery. The tool uses a deep neural network that was trained on a diverse set of global images.

Supported input includes L1C files and ACOLITE-processed L2R files. The output geotiff contains probabilities of water pixels being optically shallow and deep. 


Originally coded by by Galen Richardson and Anders Knudby, modified and uploaded by Yulun Wu


 
## Installation 

1 - Create a conda environment and activate it: 

```bash
conda create --name opticallyshallowdeep
conda activate opticallyshallowdeep
```

2 - Install tensorflow 

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


3 - Install other dependencies: 

```bash
conda install -c conda-forge geopandas rasterio
```


4 - Install opticallyshallowdeep: 

```bash
pip3 install opticallyshallowdeep
```


## Quick Start

```python
import opticallyshallowdeep as osd
file_in = 'path1'
file_out = 'path2'
osd.run(file_in, file_out)
```





