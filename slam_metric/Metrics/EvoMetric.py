import os


def eval_evo(file_path, file_name, gt_file_path, pred_file_path, traj=False, in_meter = False):
    if (os.path.isfile(os.path.join(file_path, f'{file_name}_rpe_map.jpg'))):
        os.remove(os.path.join(file_path, f'{file_name}_rpe_map.jpg'))  # delete plot file if exist
    if (os.path.isfile(os.path.join(file_path, f'{file_name}_rpe_raw.jpg'))):
        os.remove(os.path.join(file_path, f'{file_name}_rpe_raw.jpg'))  # delete plot file if exist
    command = f'evo_rpe kitti "{gt_file_path}" "{pred_file_path}" -a --plot_mode xy --save_plot "{os.path.join(file_path,file_name)}_rpe.jpg"'
    if in_meter:
        command += ' --delta 1 --delta_unit m '
    rpe_result = ParseEvo(os.popen(command).read())

    if (os.path.isfile(os.path.join(file_path, f'{file_name}_ape_map.jpg'))):
        os.remove(os.path.join(file_path, f'{file_name}_ape_map.jpg'))  # delete plot file if exist
    if (os.path.isfile(os.path.join(file_path, f'{file_name}_ape_raw.jpg'))):
        os.remove(os.path.join(file_path, f'{file_name}_ape_raw.jpg'))  # delete plot file if exist
    command = f'evo_ape kitti "{gt_file_path}" "{pred_file_path}" -a --plot_mode xy --save_plot "{os.path.join(file_path,file_name)}_ape.jpg"'
    ape_result = ParseEvo(os.popen(command).read())

    if traj:
        command = f'evo_traj kitti --silent --save_plot "{os.path.join(file_path, file_name)}" --ref="{gt_file_path}" "{pred_file_path}"'
        os.system(command)

    return rpe_result, ape_result


def ParseEvo(evo_response):
    '''
    prase response of evo system call (e.g, evo_rpe kitti xxx.txt xxx.txt -v)
    evo_response: str, result of system call
    '''
    result = dict()
    for line in evo_response.splitlines():
        line_strip = line.strip()
        if (line_strip.startswith("mean")):
            result['mean'] = float(line.split()[1])
        elif (line_strip.startswith("max")):
            result['max'] = float(line.split()[1])
        elif (line_strip.startswith("min")):
            result['min'] = float(line.split()[1])
        elif (line_strip.startswith("median")):
            result['median'] = float(line.split()[1])
        elif (line_strip.startswith("rmse")):
            result['rmse'] = float(line.split()[1])
        elif (line_strip.startswith("sse")):
            result['sse'] = float(line.split()[1])
        elif (line_strip.startswith("std")):
            result['std'] = float(line.split()[1])
    return result
