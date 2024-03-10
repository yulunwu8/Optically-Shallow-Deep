
import re



def parse_string(s):
    if s.lower() in ["lat", "long", "lat_abs"]:
        return [s.lower()]#parses the column names for our model and makes into things that our scripts can interpret
    match = re.match(r'B(\d+)([a-zA-Z]?)-w_(\d+)(_?(\w+))?', s) #used to parse model_column names
    if match:
        groups = match.groups()
        first_group = f'{groups[0]}{groups[1]}' if groups[1] else groups[0]
        first_group = first_group.replace('8a', '9')# Replace '8a' with '9'
        try:
            first_group_int = int(first_group)# Check if first_group is a valid integer
            if first_group_int > 10:
                first_group_int -= 2
            else:
                first_group_int -= 1
        except ValueError:
            first_group_int = first_group
        return [first_group_int, int(groups[2]), groups[4]] if groups[4] else [first_group_int, int(groups[2]), None]
    else:
        return None



GTOA_model_columns=['long', 'lat_abs', 'B2-w_15_sd', 'B3-w_3_sd', 'B3-w_7_avg', 'B3-w_9_avg', 'B3-w_11_sd', 'B3-w_15_sd', 'B4-w_5_avg', 'B4-w_11_sd', 'B4-w_13_avg', 'B4-w_13_sd', 'B4-w_15_sd', 'B5-w_13_sd', 'B5-w_15_sd', 'B8-w_9_sd', 'B8-w_13_sd', 'B8-w_15_sd', 'B11-w_9_avg', 'B11-w_15_sd']
GACOLITE_model_columns= ['lat_abs', 'B1-w_15_sd', 'B2-w_1', 'B3-w_5_sd', 'B3-w_7_sd', 'B3-w_13_sd', 'B3-w_15_sd', 'B4-w_15_sd', 'B5-w_11_avg', 'B5-w_15_avg', 'B5-w_15_sd', 'B8-w_13_sd', 'B11-w_15_avg']       



test = parse_string('B1-w_15_sd')


test2 = [parse_string(s) for s in GTOA_model_columns]


