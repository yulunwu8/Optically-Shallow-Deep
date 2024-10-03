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
