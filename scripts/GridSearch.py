import datetime
import os
from time import time
import tqdm
import itertools

hyperparameter = {
    '--key_frame_energy': [0.97, 0.99, 0.999, 0.9999, 0.95, 0.90, 0.8, 0.7, 0.6, ],
    '--energy_method': ['norm', 'const'],
}
parameter_str = " "
for p in hyperparameter.keys():
    parameter_str += (p + ' {' + p + '} ')
parameter_comb = list(itertools.product(*hyperparameter.values()))
print(parameter_comb)

while (input('Press Enter to continue...') != ''):
    pass

search_token = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.makedirs('./result_search', exist_ok=True)
logfile = os.path.join('result_search', search_token + '.csv')
if (os.path.isfile(logfile)):
    while (input(f'Log file {logfile} exists, type "D" to delete...') != 'D'):
        pass
    os.remove(logfile)

with open(logfile, 'a+') as f:
    f.write(', '.join([str(k) for k in hyperparameter.keys()]) + ', metrics...\n')
for comb in parameter_comb:
    format_dict = {p: v for p, v in zip(hyperparameter.keys(), comb)}
    parameter_str_run = parameter_str.format(**format_dict)
    name = parameter_str_run.strip().replace('-', '').replace(' ', '_')
    command = f"python ./Alignment_OriginPCD/main.py --mode eval --num_worker 2 --batch_size 1 --yaml_cfg ./config/xiaze.grid.yaml --mode eval --name GridSearch_Norm_{name}" + parameter_str_run
    print(command)
    start_time = time()
    result = os.popen(command).read()
    duration_time = time() - start_time

    metric = result.split('\n')[-2]
    print('================================')
    print(metric)
    print('================================')
    with open(logfile, 'a+') as f:
        f.write(', '.join([str(v) for v in format_dict.values()]) + ', ' + metric + '\n')
