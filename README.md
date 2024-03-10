# Optically-Shallow-Deep 

Package name: opticallyshallowdeep

Usage: import opticallyshallowdeep as osd

Written by Galen Richardson, modified and uploaded by Yulun Wu


 
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





