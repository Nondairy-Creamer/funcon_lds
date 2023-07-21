import loading_utilities as lu
import fit_data
import simple_slurm


param_name = 'synthetic_test'
run_params = lu.get_run_params(param_name=param_name)

if 'slurm' in run_params.keys():

else:
    method = getattr(fit_data, 'fit_synthetic')
    method(run_params)
