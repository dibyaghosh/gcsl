from datetime import datetime

import doodad
import doodad.mode
import doodad.mount as mount
from doodad.utils import REPO_DIR
from doodad.easy_sweep.hyper_sweep import run_sweep_doodad, run_sweep_parallel, run_sweep_serial

INSTANCE_TO_PRICE = {
    'c4.large': 0.03,
    'c4.xlarge': 0.05,
    'c4.2xlarge': 0.08,
    'p2.xlarge': 0.2,
}

class DoodadSweeper(object):
    def __init__(self,
            mounts,
            docker_img='python:3.5',
            docker_output_dir='/data',
            local_output_dir='data/docker',
            ):

        self.image = docker_img
        self.mode_local = doodad.mode.LocalDocker(image=docker_img)

        # always include doodad
        mounts.append(mount.MountLocal(local_dir=REPO_DIR, pythonpath=True))
        self.mounts = mounts
        self.mount_out_local = mount.MountLocal(local_dir=local_output_dir, mount_point=docker_output_dir, output=True)
        self.mount_out_s3 = mount.MountS3(s3_path='exp_logs', mount_point=docker_output_dir, output=True)

    def run_sweep_serial(self, run_method, params, repeat=1):
        run_sweep_serial(run_method, params, repeat=repeat)

    def run_sweep_parallel(self, run_method, params, repeat=1):
        run_sweep_parallel(run_method, params, repeat=repeat)

    def run_test_docker(self, run_method, params, **kwargs):
        run_sweep_doodad(run_method, params, run_mode=self.mode_local,
                         mounts=self.mounts+[self.mount_out_local],
                         test_one=True)

    def run_single_docker(self, run_method, kwargs):
        run_single_doodad(run_method, kwargs, run_mode=self.mode_local,
                         mounts=self.mounts+[self.mount_out_local])

    def run_sweep_ec2(self, run_method, params, bucket_name, 
                      s3_log_name=None, add_date_to_logname=True,
                      region='us-east-2', instance_type='c4.xlarge', repeat=1):
        if s3_log_name is None:
            s3_log_name = 'unnamed_experiment'
        if add_date_to_logname:
            datestamp = datetime.now().strftime('%Y_%m_%d')
            s3_log_name = '%s_%s' % (datestamp, s3_log_name)

        mode_ec2 = doodad.mode.EC2AutoconfigDocker(
            image=self.image,
            region=region,
            s3_bucket=bucket_name,
            instance_type=instance_type,
            spot_price=INSTANCE_TO_PRICE[instance_type],
            s3_log_prefix=s3_log_name,
        )
        run_sweep_doodad(run_method, params, run_mode=mode_ec2, 
                mounts=self.mounts+[self.mount_out_s3], repeat=repeat)

    def run_single_ec2(self, run_method, kwargs, bucket_name, 
                      s3_log_name=None, add_date_to_logname=True,
                      region='us-east-2', instance_type='c4.xlarge'):
        if s3_log_name is None:
            s3_log_name = 'unnamed_experiment'
        if add_date_to_logname:
            datestamp = datetime.now().strftime('%Y_%m_%d')
            s3_log_name = '%s_%s' % (datestamp, s3_log_name)

        mode_ec2 = doodad.mode.EC2AutoconfigDocker(
            image=self.image,
            region=region,
            s3_bucket=bucket_name,
            instance_type=instance_type,
            spot_price=INSTANCE_TO_PRICE[instance_type],
            s3_log_prefix=s3_log_name,
        )
        run_single_doodad(run_method, kwargs, run_mode=mode_ec2, 
                mounts=self.mounts+[self.mount_out_s3])

if __name__ == "__main__":
    # test
    def example_function(param1=0, param2='c'):
        print(param1, param2)
    sweep_params = {
        'param1': [0,1,2],
        'param2': ['a','b']
    }
    SWEEPER = DoodadSweeper([], docker_img='justinfu/rl_base:0.1')
    #SWEEPER.run_sweep_serial(example_function, sweep_params)
    SWEEPER.run_sweep_ec2(example_function, sweep_params, bucket_name='doodad')

