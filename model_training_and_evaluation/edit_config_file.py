import sys
import shutil


#parent_directory_for_all_output = sys.argv[1]


def change_variable_in_config_to_value(variable_name, desired_value):
    found_match = False
    try:
        float(desired_value)
    except:
        if desired_value.lower() == 'true':
            desired_value = 'True'
        elif desired_value.lower() == 'false':
            desired_value = 'False'
        else:
            if not desired_value.startswith('"') and not desired_value.startswith("'"):
                if desired_value.endswith('"'):
                    desired_value = '"' + desired_value
                elif desired_value.endswith("'"):
                    desired_value = "'" + desired_value
                else:
                    desired_value = '"' + desired_value
            if not desired_value.endswith('"') and not desired_value.endswith("'"):
                if desired_value.startswith('"'):
                    desired_value = desired_value + '"'
                elif desired_value.endswith("'"):
                    desired_value = desired_value + "'"
    with open('config.py', 'r') as f:
        with open(temp_fname, 'w') as new_f:
            for line in f:
                if line.strip().startswith('#'):
                    new_f.write(line)
                    continue
                line_parts = line.split('=')
                if line_parts[0].strip() == variable_name:
                    found_match = True
                    comment_at_end_of_line = '#' in line  # not always true, but it is in our config file
                    if comment_at_end_of_line:
                        comment_at_end_of_line = line[line.index('#'):]
                        line = line[:line.index('#')]
                        line_parts = line.split('=')
                    assert len(line_parts) == 2, line
                    if line_parts[1].startswith(' '):
                        desired_value = ' ' + desired_value
                    if comment_at_end_of_line:
                        desired_value = desired_value + '  '
                    line_parts[1] = desired_value
                    line = '='.join(line_parts)
                    if comment_at_end_of_line:
                        line = line + comment_at_end_of_line
                        if not comment_at_end_of_line.endswith('\n'):
                            line = line + '\n'
                    else:
                        line = line + '\n'
                    new_f.write(line)
                else:
                    new_f.write(line)
    assert found_match
    shutil.move(temp_fname, 'config.py')


if __name__ == '__main__':
    temp_fname = 'config_temp.py'
    varnames = []
    vals = []
    collecting_varname = True
    for i in range(1, len(list(sys.argv))):
        if collecting_varname:
            varnames.append(sys.argv[i].strip())
        else:
            vals.append(sys.argv[i].strip())
        collecting_varname = not collecting_varname
    assert len(varnames) == len(vals)
    for varname, val in zip(varnames, vals):
        change_variable_in_config_to_value(varname, val)
