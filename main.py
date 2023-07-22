import loading_utilities as lu
import fit_data
import sys
import os
from simple_slurm import Slurm
from datetime import datetime, time
from pathlib import Path

if len(sys.argv) == 1:
    param_name = 'synthetic_test'
else:
    param_name = sys.argv[1]

run_params = lu.get_run_params(param_name=param_name)
current_date = datetime.today().strftime('%Y_%m_%d')
now = datetime.now()
beginning_of_day = datetime.combine(now.date(), time(0))
seconds_of_day = (now - beginning_of_day).seconds

save_folder = 'trained_models' / Path(param_name) / Path(current_date + '_' + str(seconds_of_day))
os.makedirs(save_folder)

if 'slurm' in run_params.keys():
    # default values that should be the same for
    full_path = Path(__file__).parent.resolve()
    slurm = Slurm(**run_params['slurm'], output=full_path / save_folder)

    run_command = ['module purge',
                   'module load anaconda3/2022.10',
                   'module load openmpi/gcc/4.1.2,',
                   'conda activate fast-mpi4py,',
                   'srun python -uc \"import fit_data; fit_data.' + run_params['fit_file'] + '(\'' + param_name + '\',\'' + save_folder + '\')\"',
                   ]

    slurm.sbatch('\n'.join(run_command))

else:
    method = getattr(fit_data, run_params['fit_file'])
    method(param_name, save_folder)
