from setuptools import setup, find_packages

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name='opticallyshallowdeep',
    version='1.2.3',
    author='Yulun Wu',
    author_email='yulunwu8@gmail.com',
    description='Identify optically shallow and deep waters in satellite imagery',
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    # packages=find_packages(),
    packages=['opticallyshallowdeep','opticallyshallowdeep.models'],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.8',
    install_requires=['geopandas','rasterio==1.3.9','tifffile==2023.8.12','netCDF4','pyproj',
                      'joblib','scipy','matplotlib','imagecodecs','tensorflow']
)











