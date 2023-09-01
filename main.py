import loading_utilities as lu
import run_inference
import sys
import os
from simple_slurm import Slurm
from datetime import datetime
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
    # param_name = 'submission_scripts/infer_post.yml'
    # param_name = 'submission_scripts/slurm_test.yml'
    param_name = 'submission_scripts/exp_DL2_IL45_N80_R0.yml'
else:
    param_name = sys.argv[1]

if len(sys.argv) == 3:
    infer_post = True
    folder_name = sys.argv[2]
else:
    infer_post = False
    folder_name = None

param_name = Path(param_name)
run_params = lu.get_run_params(param_name=param_name)

if cpu_id == 0:
    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')

    full_path = Path(__file__).parent.resolve()
    if infer_post:
        save_folder = full_path / 'trained_models' / param_name.stem / folder_name
        run_params['fit_file'] = 'infer_posterior'
    else:
        save_folder = full_path / 'trained_models' / param_name.stem / current_date
        os.makedirs(save_folder)

else:
    save_folder = None

if 'slurm' in run_params.keys():
    if cpu_id == 0:
        if infer_post:
            run_params['slurm']['time'] = '24:00:00'
            slurm_output_path = save_folder / 'slurm_%A_post.out'
            job_name = param_name.stem + '_post'
        else:
            slurm_output_path = save_folder / 'slurm_%A.out'
            job_name = param_name.stem

        if 'mem_per_task' in run_params['slurm']:
            run_params['slurm']['mem_per_cpu'] = str(int(run_params['slurm']['mem_per_task'] / run_params['slurm']['cpus_per_task'])) + 'G'
            del run_params['slurm']['mem_per_task']

        slurm_fit = Slurm(**run_params['slurm'], output=slurm_output_path, job_name=job_name)

        cpus_per_task = run_params['slurm']['cpus_per_task']
        fit_model_command = 'run_inference.' + run_params['fit_file'] + '(\'' + str(param_name) + '\',\'' + str(save_folder) + '\')\"'

        run_command = ['module purge',
                       'module load anaconda3/2022.10',
                       'module load openmpi/gcc/4.1.2',
                       'conda activate fast-mpi4py',
                       'export MKL_NUM_THREADS=' + str(cpus_per_task),
                       'export OPENBLAS_NUM_THREADS=' + str(cpus_per_task),
                       'export OMP_NUM_THREADS=' + str(cpus_per_task),
                       'srun python -uc \"import run_inference; ' + fit_model_command,
                       ]

        slurm_fit.sbatch('\n'.join(run_command))

else:
    method = getattr(run_inference, run_params['fit_file'])
    method(param_name, save_folder)
