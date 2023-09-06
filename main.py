import loading_utilities as lu
import run_inference
import sys
import os
from simple_slurm import Slurm
from datetime import datetime
from pathlib import Path
from mpi4py import MPI
from mpi4py.util import pkl5


def main(param_name, folder_name=None):
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    cpu_id = comm.Get_rank()

    if folder_name is None:
        infer_post = False
    else:
        infer_post = True

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

    return save_folder


if __name__ == '__main__':
    num_args = len(sys.argv)

    if num_args == 1:
        param_name = 'submission_scripts/syn_test.yml'
        # param_name = 'submission_scripts/exp_test.yml'
        folder_name = None
    elif num_args == 2:
        param_name = sys.argv[1]
        folder_name = None
    elif num_args == 3:
        param_name = sys.argv[1]
        folder_name = sys.argv[2]
    else:
        raise Exception('Unsupported number of arguments: (' + str(num_args))

    main(param_name, folder_name)

