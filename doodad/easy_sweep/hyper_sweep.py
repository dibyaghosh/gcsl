"""
Usage

args = {
'param1': [1e-3, 1e-2, 1e-2],
'param2': [1,5,10,20],
}

run_sweep_parallel(func, args)

or

run_sweep_serial(func, args)

"""
import os
import itertools
import multiprocessing
import random
from datetime import datetime

import doodad
from doodad.utils import REPO_DIR


class Sweeper(object):
    def __init__(self, hyper_config, repeat, include_name=False):
        self.hyper_config = hyper_config
        self.repeat = repeat
        self.include_name=include_name

    def __iter__(self):
        count = 0
        for _ in range(self.repeat):
            for config in itertools.product(*[val for val in self.hyper_config.values()]):
                kwargs = {key:config[i] for i, key in enumerate(self.hyper_config.keys())}
                if self.include_name:
                    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                    kwargs['exp_name'] = "%s_%d" % (timestamp, count)
                count += 1
                yield kwargs


def run_sweep_serial(run_method, params, repeat=1):
    sweeper = Sweeper(params, repeat)
    for config in sweeper:
        run_method(**config)


def kwargs_wrapper(args_method):
    args, method = args_method
    return method(**args)


def run_sweep_parallel(run_method, params, repeat=1, num_cpu=multiprocessing.cpu_count()):
    sweeper = Sweeper(params, repeat)
    pool = multiprocessing.Pool(num_cpu)
    exp_args = []
    for config in sweeper:
        exp_args.append((config, run_method))
    random.shuffle(exp_args)
    pool.map(kwargs_wrapper, exp_args)


SCRIPTS_DIR = os.path.join(REPO_DIR, 'scripts')
def run_sweep_doodad(run_method, params, run_mode, mounts, repeat=1, test_one=False):
    sweeper = Sweeper(params, repeat)
    for config in sweeper:
        def run_method_args():
            run_method(**config)
        doodad.launch_python(
                target = os.path.join(SCRIPTS_DIR, 'run_experiment_lite_doodad.py'),
                mode=run_mode,
                mount_points=mounts,
                use_cloudpickle=True,
                args = {'run_method': run_method_args},
        )
        if test_one:
            break


def run_single_doodad(run_method, kwargs, run_mode, mounts, repeat=1):
    """ Run a single function via doodad """
    sweeper = Sweeper(params, repeat)
    def run_method_args():
        run_method(**kwargs)
    doodad.launch_python(
            target = os.path.join(SCRIPTS_DIR, 'run_experiment_lite_doodad.py'),
            mode=run_mode,
            mount_points=mounts,
            use_cloudpickle=True,
            args = {'run_method': run_method_args},
    )


if __name__ == "__main__":
    def example_run_method(exp_name, param1, param2='a', param3=3, param4=4):
        import time
        time.sleep(1.0)
        print(exp_name, param1, param2, param3, param4)
    sweep_op = {
        'param1': [1e-3, 1e-2, 1e-1],
        'param2': [1,5,10,20],
        'param3': [True, False]
    }
    run_sweep_parallel(example_run_method, sweep_op, repeat=2)
