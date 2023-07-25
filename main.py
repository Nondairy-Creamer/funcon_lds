import loading_utilities as lu
import fit_data
import sys
import os
from simple_slurm import Slurm
from datetime import datetime, time
from pathlib import Path
from mpi4py import MPI
from mpi4py.util import pkl5

comm = pkl5.Intracomm(MPI.COMM_WORLD)
size = comm.Get_size()
rank = comm.Get_rank()
is_parallel = size > 1

if len(sys.argv) == 1:
    # param_name = 'syn_test'
    param_name = 'exp_test'
else:
    param_name = sys.argv[1]

run_params = lu.get_run_params(param_name=param_name)

if rank == 0:
    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
    now = datetime.now()
    beginning_of_day = datetime.combine(now.date(), time(0))
    seconds_of_day = (now - beginning_of_day).seconds

    full_path = Path(__file__).parent.resolve()
    save_folder = full_path / 'trained_models' / Path(param_name) / Path(current_date + '_' + str(seconds_of_day))
    os.makedirs(save_folder)
else:
    save_folder = None

if 'slurm' in run_params.keys():
    if rank == 0:
        # default values that should be the same for
        slurm = Slurm(**run_params['slurm'], output=save_folder/'slurm_%A.out', job_name=param_name)

        run_command = ['module purge',
                       'module load anaconda3/2022.10',
                       'module load openmpi/gcc/4.1.2',
                       'conda activate fast-mpi4py',
                       'srun python -uc \"import fit_data; fit_data.' + run_params['fit_file'] + '(\'' + str(param_name) + '\',\'' + str(save_folder) + '\')\"',
                       ]

        slurm.sbatch('\n'.join(run_command))

else:
    method = getattr(fit_data, run_params['fit_file'])
    method(param_name, save_folder)
