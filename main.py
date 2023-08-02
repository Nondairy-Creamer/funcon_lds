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
cpu_id = comm.Get_rank()
is_parallel = size > 1

if len(sys.argv) == 1:
    # param_name = 'submission_scripts/syn_test.yml'
    # param_name = 'submission_scripts/exp_test.yml'
    param_name = 'submission_scripts/infer_test_data.yml'
else:
    param_name = sys.argv[1]

param_name = Path(param_name)
run_params = lu.get_run_params(param_name=param_name)

if cpu_id == 0:
    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')

    full_path = Path(__file__).parent.resolve()
    save_folder = full_path / 'trained_models' / param_name.stem / current_date
    os.makedirs(save_folder)
else:
    save_folder = None

if 'slurm' in run_params.keys():
    if cpu_id == 0:
        # default values that should be the same for
        slurm = Slurm(**run_params['slurm'], output=save_folder/'slurm_%A.out', job_name=param_name.stem)

        cpus_per_task = run_params['slurm']['cpus_per_task']

        run_command = ['module purge',
                       'module load anaconda3/2022.10',
                       'module load openmpi/gcc/4.1.2',
                       'conda activate fast-mpi4py',
                       'export MKL_NUM_THREADS=' + str(cpus_per_task),
                       'export OPENBLAS_NUM_THREADS=' + str(cpus_per_task),
                       'export OMP_NUM_THREADS=' + str(cpus_per_task),
                       'srun python -uc \"import fit_data; fit_data.' + run_params['fit_file'] + '(\'' + str(param_name) + '\',\'' + str(save_folder) + '\')\"',
                       ]

        slurm.sbatch('\n'.join(run_command))

else:
    method = getattr(fit_data, run_params['fit_file'])
    method(param_name, save_folder)
