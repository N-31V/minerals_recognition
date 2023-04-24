import argparse
from datetime import datetime

from fedot_ind.core.operation.optimization.structure_optimization import SFPOptimization, SVDOptimization
from minerals_config import TASKS, EXPS


def run_base(at, ad):
    task = TASKS[at](ad)
    task['exp'].fit(p=task['fit_params'])


def run_svd_c(at, ad, h=0.1, o=10):
    task = TASKS[at](ad)
    svd_optim = SVDOptimization(
        energy_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9, 0.93, 0.96, 0.99, 0.999],
        hoer_loss_factor=h,
        orthogonal_loss_factor=o,
    )
    svd_optim.fit(task['exp'], params=task['fit_params'], ft_params=task['ft_params'])


def run_svd_s(at, ad, h=0.1, o=10):
    task = TASKS[at](ad)
    svd_optim = SVDOptimization(
        energy_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9, 0.93, 0.96, 0.99, 0.999],
        decomposing_mode='spatial',
        hoer_loss_factor=h,
        orthogonal_loss_factor=o,
    )
    svd_optim.fit(task['exp'], params=task['fit_params'], ft_params=task['ft_params'])


def run_sfp_p(at, ad, p=0.1):
    task = TASKS[at](ad)
    sfp_optim = SFPOptimization(
        zeroing_mode='percentage',
        zeroing_mode_params={'pruning_ratio': p}
    )
    sfp_optim.fit(task['exp'], params=task['fit_params'], ft_params=task['ft_params'])


def run_sfp_e(at, ad, e=0.99):
    task = TASKS[at](ad)
    sfp_optim = SFPOptimization(
        zeroing_mode='energy',
        zeroing_mode_params={'energy_threshold': e}
    )
    sfp_optim.fit(task['exp'], params=task['fit_params'], ft_params=task['ft_params'])


OPTIM = {
    'base': run_base,
    'svd_c': run_svd_c,
    'svd_s': run_svd_s,
    'sfp_p': run_sfp_p,
    'sfp_e': run_sfp_e,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Minerals')
    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-e', '--experiments', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, required=True)
    args = parser.parse_args()

    start_t = datetime.now()
    for opt, params in EXPS[args.experiments]:
        OPTIM[opt](args.task, args.device, **params)
    print(f'Total time: {datetime.now() - start_t}')
