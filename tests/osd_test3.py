



def find_epsg(data):
    
    
    # Use regex to extract the desired part
    import re
    from pyproj import CRS
    
    # Pattern to match the text between "CONVERSION[" and the first comma following
    pattern = r'CONVERSION\["(.*?)",'
    
    # Search for the pattern in the data string
    match = re.search(pattern, data)
    
    # Extract the matched text
    extracted_text = match.group(1) if match else None
    
    extracted_text
    
    
    # Define the CRS name
    # crs_name = "WGS 84 / UTM zone 15N"
    
    crs_name = "WGS 84 / " + str(extracted_text)
    
    
    try:
        # Use pyproj to find the CRS object from the given CRS name
        crs = CRS.from_string(crs_name)
        # Extract the EPSG code
        epsg_code = crs.to_epsg()
        if epsg_code:
            # return f"The EPSG code for '{crs_name}' is {epsg_code}."
            return epsg_code
        
        
        else:
            print( "EPSG code could not be found.")
        
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

data = 'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ID["EPSG",6326]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]]],CONVERSION["UTM zone 15N",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",-93,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]],ID["EPSG",16015]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]'

# Call the function and print the result
result = find_epsg(data)
print(result)



