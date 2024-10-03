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