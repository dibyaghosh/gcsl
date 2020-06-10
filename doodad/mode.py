import os
import subprocess
import tempfile
import uuid
import time
import base64

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from .mount import MountLocal, MountS3
from .utils import hash_file, call_and_wait, CommandBuilder
from .ec2.aws_util import s3_upload, s3_exists


class LaunchMode(object):
    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        raise NotImplementedError()


class Local(LaunchMode):
    def __init__(self):
        super(Local, self).__init__()
        self.env = {}

    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        if dry:
            print(cmd); return

        commands = CommandBuilder()
        # chdir to home dir
        commands.append('cd %s' % (os.path.expanduser('~')))

        # do mounting
        py_path = []
        cleanup_commands = CommandBuilder()
        for mount in mount_points:
            print('mounting:', mount)
            if isinstance(mount, MountLocal):
                if not mount.no_remount:
                    mount.create_if_nonexistent()
                    commands.append('ln -s %s %s' % (mount.local_dir, mount.mount_point))
                    #subprocess.call(symlink_cmd, shell=True)
                    if mount.cleanup:
                        cleanup_commands.append('rm "%s"' % mount.mount_point)
                if mount.pythonpath:
                    py_path.append(mount.mount_point)
            else:
                raise NotImplementedError()

        # add pythonpath mounts
        if py_path:
            commands.append('export PYTHONPATH=$PYTHONPATH:%s' % (':'.join(py_path)))

        # Add main command
        commands.append(cmd)

        # cleanup
        commands.extend(cleanup_commands)

        # Call everything
        commands.call_and_wait()

LOCAL = Local()


class DockerMode(LaunchMode):
    def __init__(self, image='ubuntu:16.04', gpu=False):
        super(DockerMode, self).__init__()
        self.docker_image = image
        self.docker_name = uuid.uuid4()
        self.gpu = gpu

    def get_docker_cmd(self, main_cmd, extra_args='', use_tty=True, verbose=True, pythonpath=None, pre_cmd=None, post_cmd=None,
            checkpoint=False, no_root=False):
        cmd_list= CommandBuilder()
        if pre_cmd:
            cmd_list.extend(pre_cmd)

        if verbose:
            if self.gpu:
                cmd_list.append('echo \"Running in docker (gpu)\"')
            else:
                cmd_list.append('echo \"Running in docker\"')
        if pythonpath:
            cmd_list.append('export PYTHONPATH=$PYTHONPATH:%s' % (':'.join(pythonpath)))
        if no_root:
            # This should work if you're running a script
            #cmd_list.append('useradd --uid $(id -u) --no-create-home --home-dir / doodaduser')
            #cmd_list.append('su - doodaduser /bin/bash {script}')

            # this is a temp workaround
            extra_args += ' -u $(id -u)'

        cmd_list.append(main_cmd)
        if post_cmd:
            cmd_list.extend(post_cmd)

        docker_name = self.docker_name
        if docker_name:
            extra_args += ' --name %s '%docker_name

        if checkpoint:
            # set up checkpoint stuff
            use_tty = False
            extra_args += ' -d '  # detach is optional

        if use_tty:
            docker_prefix = 'docker run %s -ti %s /bin/bash -c ' % (extra_args, self.docker_image)
        else:
            docker_prefix = 'docker run %s %s /bin/bash -c ' % (extra_args, self.docker_image)
        if self.gpu:
            docker_prefix = 'nvidia-'+docker_prefix
        main_cmd = cmd_list.to_string()
        full_cmd = docker_prefix + ("\'%s\'" % main_cmd)
        return full_cmd


class LocalDocker(DockerMode):
    def __init__(self, checkpoints=None, **kwargs):
        super(LocalDocker, self).__init__(**kwargs)
        self.checkpoints = checkpoints

    def launch_command(self, cmd, mount_points=None, dry=False, verbose=False):
        mnt_args = ''
        py_path = []
        for mount in mount_points:
            if isinstance(mount, MountLocal):
                #mount_pnt = os.path.expanduser(mount.mount_point)
                mount_pnt = mount.mount_dir()
                mnt_args += ' -v %s:%s' % (mount.local_dir, mount_pnt)
                call_and_wait('mkdir -p %s' % mount.local_dir)
                if mount.pythonpath:
                    py_path.append(mount_pnt)
            else:
                raise NotImplementedError(type(mount))

        full_cmd = self.get_docker_cmd(cmd, extra_args=mnt_args, pythonpath=py_path,
                checkpoint=self.checkpoints)
        if verbose:
            print(full_cmd)
        call_and_wait(full_cmd, dry=dry)


class SSHDocker(DockerMode):
    TMP_DIR = '~/.remote_tmp'

    def __init__(self, credentials=None, **docker_args):
        super(SSHDocker, self).__init__(**docker_args)
        self.credentials = credentials
        self.run_id = 'run_%s' % uuid.uuid4()
        self.tmp_dir = os.path.join(SSHDocker.TMP_DIR, self.run_id)
        self.checkpoint = None

    def launch_command(self, main_cmd, mount_points=None, dry=False, verbose=False):
        py_path = []
        remote_cmds = CommandBuilder()
        remote_cleanup_commands = CommandBuilder()
        mnt_args = ''

        tmp_dir_cmd = 'mkdir -p %s' % self.tmp_dir
        tmp_dir_cmd = self.credentials.get_ssh_bash_cmd(tmp_dir_cmd)
        call_and_wait(tmp_dir_cmd, dry=dry, verbose=verbose)

        # SCP Code over
        for mount in mount_points:
            if isinstance(mount, MountLocal):
                if mount.read_only:
                    with mount.gzip() as gzip_file:
                        # scp
                        base_name = os.path.basename(gzip_file)
                        #file_hash = hash_file(gzip_path)  # TODO: store all code in a special "caches" folder
                        remote_mnt_dir = os.path.join(self.tmp_dir, os.path.splitext(base_name)[0])
                        remote_tar = os.path.join(self.tmp_dir, base_name)
                        scp_cmd = self.credentials.get_scp_cmd(gzip_file, remote_tar)
                        call_and_wait(scp_cmd, dry=dry, verbose=verbose)
                    remote_cmds.append('mkdir -p %s' % remote_mnt_dir)
                    unzip_cmd = 'tar -xf %s -C %s' % (remote_tar, remote_mnt_dir)
                    remote_cmds.append(unzip_cmd)
                    mount_point = mount.mount_dir()
                    mnt_args += ' -v %s:%s' % (os.path.join(remote_mnt_dir, os.path.basename(mount.mount_point)) ,mount_point)
                else:
                    #remote_cmds.append('mkdir -p %s' % mount.mount_point)
                    remote_cmds.append('mkdir -p %s' % mount.local_dir_raw)
                    mnt_args += ' -v %s:%s' % (mount.local_dir_raw, mount.mount_point)

                if mount.pythonpath:
                    py_path.append(mount_point)
            else:
                raise NotImplementedError()

        if self.checkpoint and self.checkpoint.restore:
            raise NotImplementedError()
        else:
            docker_cmd = self.get_docker_cmd(main_cmd, use_tty=False, extra_args=mnt_args, pythonpath=py_path)


        remote_cmds.append(docker_cmd)
        remote_cmds.extend(remote_cleanup_commands)

        with tempfile.NamedTemporaryFile('w+', suffix='.sh') as ntf:
            for cmd in remote_cmds:
                if verbose:
                    ntf.write('echo "%s$ %s"\n' % (self.credentials.user_host, cmd))
                ntf.write(cmd+'\n')
            ntf.seek(0)
            ssh_cmd = self.credentials.get_ssh_script_cmd(ntf.name)

            call_and_wait(ssh_cmd, dry=dry, verbose=verbose)


def dedent(s):
    lines = [l.strip() for l in s.split('\n')]
    return '\n'.join(lines)

class EC2SpotDocker(DockerMode):
    def __init__(self,
            credentials,
            region='us-west-1',
            s3_bucket_region='us-west-1',
            instance_type='m1.small',
            spot_price=0.0,
            s3_bucket=None,
            terminate=True,
            image_id=None,
            aws_key_name=None,
            iam_instance_profile_name='doodad',
            s3_log_prefix='experiment',
            s3_log_name=None,
            security_group_ids=None,
            security_groups=None,
            aws_s3_path=None,
            extra_ec2_instance_kwargs=None,
            **kwargs
            ):
        super(EC2SpotDocker, self).__init__(**kwargs)
        if security_group_ids is None:
            security_group_ids = []
        if security_groups is None:
            security_groups = []
        self.credentials = credentials
        self.region = region
        self.s3_bucket_region = s3_bucket_region
        self.spot_price = str(float(spot_price))
        self.instance_type = instance_type
        self.terminate = terminate
        self.s3_bucket = s3_bucket
        self.image_id = image_id
        self.aws_key_name = aws_key_name
        self.s3_log_prefix = s3_log_prefix
        self.s3_log_name = s3_log_name
        self.security_group_ids = security_group_ids
        self.security_groups = security_groups
        self.iam_instance_profile_name = iam_instance_profile_name
        self.extra_ec2_instance_kwargs = extra_ec2_instance_kwargs
        self.checkpoint = None

        self.s3_mount_path = 's3://%s/doodad/mount' % self.s3_bucket
        self.aws_s3_path = aws_s3_path or 's3://%s/doodad/logs' % self.s3_bucket

    def upload_file_to_s3(self, script_content, dry=False):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(script_content.encode())
        f.close()
        remote_path = os.path.join(self.s3_mount_path, 'oversize_bash_scripts', str(uuid.uuid4()))
        subprocess.check_call(["aws", "s3", "cp", f.name, remote_path,
                               '--region', self.s3_bucket_region])
        os.unlink(f.name)
        return remote_path

    def s3_upload(self, file_name, bucket, remote_filename=None, dry=False, check_exist=True):
        if remote_filename is None:
            remote_filename = os.path.basename(file_name)
        remote_path = 'doodad/mount/'+remote_filename
        if check_exist:
            if s3_exists(bucket, remote_path, region=self.s3_bucket_region):
                print('\t%s exists! ' % os.path.join(bucket, remote_path))
                return 's3://'+os.path.join(bucket, remote_path)
        return s3_upload(file_name, bucket, remote_path, dry=dry,
                         region=self.s3_bucket_region)

    def make_timekey(self):
        return '%d'%(int(time.time()*1000))

    def launch_command(self, main_cmd, mount_points=None, dry=False, verbose=False):
        default_config = dict(
            image_id=self.image_id,
            instance_type=self.instance_type,
            key_name=self.aws_key_name,
            spot_price=self.spot_price,
            iam_instance_profile_name=self.iam_instance_profile_name,
            security_groups=self.security_groups,
            security_group_ids=self.security_group_ids,
            network_interfaces=[],
        )
        aws_config = dict(default_config)
        if self.s3_log_name is None:
            exp_name = "{}-{}".format(self.s3_log_prefix, self.make_timekey())
        else:
            exp_name = self.s3_log_name
        exp_prefix = self.s3_log_prefix
        s3_base_dir = os.path.join(self.aws_s3_path, exp_prefix.replace("_", "-"), exp_name)

        sio = StringIO()
        sio.write("#!/bin/bash\n")
        sio.write("truncate -s 0 /home/ubuntu/user_data.log\n")
        sio.write("{\n")
        sio.write('die() { status=$1; shift; echo "FATAL: $*"; exit $status; }\n')
        sio.write('EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`"\n')
        sio.write("""
            aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value={exp_name} --region {aws_region}
        """.format(exp_name=exp_name, aws_region=self.region))
        sio.write("""
            aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=exp_prefix,Value={exp_prefix} --region {aws_region}
        """.format(exp_prefix=exp_prefix, aws_region=self.region))
        sio.write("service docker start\n")
        sio.write("docker --config /home/ubuntu/.docker pull {docker_image}\n".format(docker_image=self.docker_image))
        sio.write("export AWS_DEFAULT_REGION={aws_region}\n".format(aws_region=self.s3_bucket_region))
        sio.write("""
            curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
            unzip awscli-bundle.zip
            sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws
        """)

        mnt_args = ''
        py_path = []
        local_output_dir_and_s3_path = []
        max_sync_interval = 0
        for mount in mount_points:
            print('Handling mount: ', mount)
            if isinstance(mount, MountLocal):  # TODO: these should be mount_s3 objects
                if mount.read_only:
                    if mount.path_on_remote is None:
                        with mount.gzip() as gzip_file:
                            gzip_path = os.path.realpath(gzip_file)
                            file_hash = hash_file(gzip_path)
                            s3_path = self.s3_upload(gzip_path, self.s3_bucket, remote_filename=file_hash+'.tar')
                        mount.path_on_remote = s3_path
                        mount.local_file_hash = gzip_path
                    else:
                        file_hash = mount.local_file_hash
                        s3_path = mount.path_on_remote
                    remote_tar_name = '/tmp/'+file_hash+'.tar'
                    remote_unpack_name = '/tmp/'+file_hash
                    sio.write("aws s3 cp {s3_path} {remote_tar_name}\n".format(s3_path=s3_path, remote_tar_name=remote_tar_name))
                    sio.write("mkdir -p {local_code_path}\n".format(local_code_path=remote_unpack_name))
                    sio.write("tar -xvf {remote_tar_name} -C {local_code_path}\n".format(
                        local_code_path=remote_unpack_name,
                        remote_tar_name=remote_tar_name))
                    mount_point =  os.path.join('/mounts', mount.mount_point.replace('~/',''))
                    mnt_args += ' -v %s:%s' % (os.path.join(remote_unpack_name, os.path.basename(mount.local_dir)), mount_point)
                    if mount.pythonpath:
                        py_path.append(mount_point)
                else:
                    raise ValueError()
            elif isinstance(mount, MountS3):
                # In theory the ec2_local_dir could be some random directory,
                # but we make it the same as the mount directory for
                # convenience.
                #
                # ec2_local_dir: directory visible to ec2 spot instance
                # moint_point: directory visible to docker running inside ec2
                #               spot instance
                ec2_local_dir = mount.mount_point
                s3_path = os.path.join(s3_base_dir, mount.s3_path)
                if not mount.output:
                    raise NotImplementedError()
                local_output_dir_and_s3_path.append(
                    (ec2_local_dir, s3_path)
                )
                sio.write("mkdir -p {remote_dir}\n".format(
                    remote_dir=ec2_local_dir)
                )
                mnt_args += ' -v %s:%s' % (ec2_local_dir, mount.mount_point)

                # Sync interval
                sio.write("""
                while /bin/true; do
                    aws s3 sync --exclude '*' {include_string} {log_dir} {s3_path}
                    sleep {periodic_sync_interval}
                done & echo sync initiated
                """.format(
                    include_string=mount.include_string,
                    log_dir=ec2_local_dir,
                    s3_path=s3_path,
                    periodic_sync_interval=mount.sync_interval
                ))
                max_sync_interval = max(max_sync_interval, mount.sync_interval)

                # Sync on terminate. This catches the case where the spot
                # instance gets terminated before the user script ends.
                #
                # This is hoping that there's at least 3 seconds between when
                # the spot instance gets marked for  termination and when it
                # actually terminates.
                sio.write("""
                    while /bin/true; do
                        if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
                        then
                            logger "Running shutdown hook."
                            aws s3 cp --recursive {log_dir} {s3_path}
                            break
                        else
                            # Spot instance not yet marked for termination.
                            # This is hoping that there's at least 3 seconds
                            # between when the spot instance gets marked for
                            # termination and when it actually terminates.
                            sleep 3
                        fi
                    done & echo log sync initiated
                """.format(
                    log_dir=ec2_local_dir,
                    s3_path=s3_path,
                ))
            else:
                raise NotImplementedError()


        sio.write("aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value={exp_name} --region {aws_region}\n".format(
            exp_name=exp_name, aws_region=self.region))

        if self.gpu:
            #sio.write('echo "LSMOD NVIDIA:"\n')
            #sio.write("lsmod | grep nvidia\n")
            #sio.write("echo 'Waiting for dpkg lock...'\n")
            # wait for lock
            #sio.write("""
            #    while sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
            #       sleep 1
            #    done
            #""")
            #sio.write("sudo apt-get install nvidia-modprobe\n")
            #sio.write("wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb\n")
            #sio.write("sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb\n")
            sio.write("""
                for i in {1..800}; do su -c "nvidia-modprobe -u -c=0" ubuntu && break || sleep 3; done
                systemctl start nvidia-docker
            """)
            sio.write("echo 'Testing nvidia-smi'\n")
            sio.write("nvidia-smi\n")
            sio.write("echo 'Testing nvidia-smi inside docker'\n")
            sio.write("nvidia-docker run --rm {docker_image} nvidia-smi\n".format(docker_image=self.docker_image))

        if self.checkpoint and self.checkpoint.restore:
            raise NotImplementedError()
        else:
            docker_cmd = self.get_docker_cmd(main_cmd, use_tty=False, extra_args=mnt_args, pythonpath=py_path)
        sio.write(docker_cmd+'\n')

        # Sync all output mounts to s3 after running the user script
        # Ideally the earlier while loop would be sufficient, but it might be
        # the case that the earlier while loop isn't fast enough to catch a
        # termination. So, we explicitly sync on termination.
        for (local_output_dir, s3_dir_path) in local_output_dir_and_s3_path:
            sio.write("aws s3 cp --recursive {local_dir} {s3_dir}\n".format(
                local_dir=local_output_dir,
                s3_dir=s3_dir_path
            ))
        sio.write("aws s3 cp /home/ubuntu/user_data.log {s3_dir_path}/stdout.log\n".format(s3_dir_path=s3_base_dir))

        # Wait for last sync
        if max_sync_interval > 0:
            sio.write("sleep {}\n".format(max_sync_interval + 5))

        if self.terminate:
            sio.write("""
                EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id || die \"wget instance-id has failed: $?\"`"
                aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID --region {aws_region}
            """.format(aws_region=self.region))
        sio.write("} >> /home/ubuntu/user_data.log 2>&1\n")

        full_script = dedent(sio.getvalue())
        import boto3
        import botocore
        ec2 = boto3.client(
            "ec2",
            region_name=self.region,
            aws_access_key_id=self.credentials.aws_key,
            aws_secret_access_key=self.credentials.aws_secret_key,
        )

        if len(full_script) > 10000 or len(base64.b64encode(full_script.encode()).decode("utf-8")) > 10000:
            s3_path = self.upload_file_to_s3(full_script, dry=dry)
            sio = StringIO()
            sio.write("#!/bin/bash\n")
            sio.write("""
            aws s3 cp {s3_path} /home/ubuntu/remote_script.sh --region {aws_region} && \\
            chmod +x /home/ubuntu/remote_script.sh && \\
            bash /home/ubuntu/remote_script.sh
            """.format(s3_path=s3_path, aws_region=self.s3_bucket_region))
            user_data = dedent(sio.getvalue())
        else:
            user_data = full_script

        if verbose:
            print(full_script)
            with open("/tmp/full_ec2_script", "w") as f:
                f.write(full_script)

        instance_args = dict(
            ImageId=aws_config["image_id"],
            KeyName=aws_config["key_name"],
            UserData=user_data,
            InstanceType=aws_config["instance_type"],
            EbsOptimized=False,
            SecurityGroups=aws_config["security_groups"],
            SecurityGroupIds=aws_config["security_group_ids"],
            NetworkInterfaces=aws_config["network_interfaces"],
            IamInstanceProfile=dict(
                Name=aws_config["iam_instance_profile_name"],
            ),
            #**config.AWS_EXTRA_CONFIGS,
        )
        if self.extra_ec2_instance_kwargs is not None:
            instance_args.update(self.extra_ec2_instance_kwargs)

        if verbose:
            print("************************************************************")
            print('UserData:', instance_args["UserData"])
            print("************************************************************")
        instance_args["UserData"] = base64.b64encode(instance_args["UserData"].encode()).decode("utf-8")
        spot_args = dict(
            DryRun=dry,
            InstanceCount=1,
            LaunchSpecification=instance_args,
            SpotPrice=aws_config["spot_price"],
            # ClientToken=params_list[0]["exp_name"],
        )

        import pprint

        if verbose:
            pprint.pprint(spot_args)
        if not dry:
            response = ec2.request_spot_instances(**spot_args)
            print('Launched EC2 job - Server response:')
            pprint.pprint(response)
            print('*****'*5)
            spot_request_id = response['SpotInstanceRequests'][
                0]['SpotInstanceRequestId']
            for _ in range(10):
                try:
                    ec2.create_tags(
                        Resources=[spot_request_id],
                        Tags=[
                            {'Key': 'Name', 'Value': exp_name}
                        ],
                    )
                    break
                except botocore.exceptions.ClientError:
                    continue


class EC2AutoconfigDocker(EC2SpotDocker):
    def __init__(self,
            region='us-west-1',
            s3_bucket=None,
            image_id=None,
            aws_key_name=None,
            iam_profile=None,
            **kwargs
            ):
        # find config file
        from doodad.ec2.autoconfig import AUTOCONFIG
        from doodad.ec2.credentials import AWSCredentials
        s3_bucket = AUTOCONFIG.s3_bucket() if s3_bucket is None else s3_bucket
        image_id = AUTOCONFIG.aws_image_id(region) if image_id is None else image_id
        aws_key_name= AUTOCONFIG.aws_key_name(region) if aws_key_name is None else aws_key_name
        iam_profile= AUTOCONFIG.iam_profile_name() if iam_profile is None else iam_profile
        credentials=AWSCredentials(aws_key=AUTOCONFIG.aws_access_key(), aws_secret=AUTOCONFIG.aws_access_secret())
        security_group_ids = AUTOCONFIG.aws_security_group_ids()[region]
        security_groups = AUTOCONFIG.aws_security_groups()

        super(EC2AutoconfigDocker, self).__init__(
                s3_bucket=s3_bucket,
                image_id=image_id,
                aws_key_name=aws_key_name,
                iam_instance_profile_name=iam_profile,
                credentials=credentials,
                region=region,
                security_groups=security_groups,
                security_group_ids=security_group_ids,
                **kwargs
                )


class CodalabDocker(DockerMode):
    def __init__(self):
        super(CodalabDocker, self).__init__()
        raise NotImplementedError()


class SingularityMode(LaunchMode):
    def __init__(self, image, gpu=False):
        super(SingularityMode, self).__init__()
        self.singularity_image = image
        self.gpu = gpu

    def get_singularity_cmd(
            self,
            main_cmd,
            extra_args='',
            verbose=True,
            pythonpath=None,
            pre_cmd=None,
            post_cmd=None,
        ):
        cmd_list= CommandBuilder()
        if pre_cmd:
            cmd_list.extend(pre_cmd)

        if verbose:
            if self.gpu:
                cmd_list.append('echo \"Running in singularity (gpu)\"')
            else:
                cmd_list.append('echo \"Running in singularity\"')
        if pythonpath:
            cmd_list.append('export PYTHONPATH=$PYTHONPATH:%s' % (':'.join(pythonpath)))

        cmd_list.append(main_cmd)
        if post_cmd:
            cmd_list.extend(post_cmd)

        if self.gpu:
            extra_args += ' --nv '
        singularity_prefix = 'singularity exec %s %s /bin/bash -c ' % (
                extra_args,
                self.singularity_image,
        )
        main_cmd = cmd_list.to_string()
        full_cmd = singularity_prefix + ("\'%s\'" % main_cmd)
        return full_cmd


class LocalSingularity(SingularityMode):
    def launch_command(self, cmd, mount_points=None, dry=False,
                       verbose=False, pre_cmd=None, post_cmd=None):
        py_path = []
        for mount in mount_points:
            if isinstance(mount, MountLocal):
                if mount.pythonpath:
                    py_path.append(mount.local_dir)
            else:
                raise NotImplementedError(type(mount))

        full_cmd = self.get_singularity_cmd(
            cmd,
            pythonpath=py_path,
            pre_cmd=pre_cmd,
            post_cmd=post_cmd,
            verbose=verbose,
        )
        if verbose:
            print(full_cmd)
        call_and_wait(full_cmd, dry=dry)


class SlurmSingularity(LocalSingularity):
    # TODO: set up an auto-config
    def __init__(
        self, image, account_name, partition, time_in_mins,
        qos=None,
        nodes=1,
        n_tasks=1,
        n_gpus=1,
        **kwargs
    ):
        super(SlurmSingularity, self).__init__(image, **kwargs)
        self.account_name = account_name
        self.partition = partition
        self.time_in_mins = time_in_mins
        self.nodes = nodes
        self.n_tasks = n_tasks
        self.n_gpus = n_gpus

    def launch_command(self, cmd, mount_points=None, dry=False,
                       verbose=False, pre_cmd=None, post_cmd=None):
        if pre_cmd is None:
            pre_cmd = []
        py_path = []
        for mount in mount_points:
            if isinstance(mount, MountLocal):
                if mount.pythonpath:
                    py_path.append(mount.local_dir)
            else:
                raise NotImplementedError(type(mount))

        singularity_cmd = self.get_singularity_cmd(
            cmd,
            pythonpath=py_path,
            pre_cmd=pre_cmd,
            post_cmd=post_cmd,
            verbose=verbose,
        )
        if self.gpu:
            full_cmd = (
                "sbatch -A {account_name} -p {partition} -t {time}"
                " -N {nodes} -n {n_tasks} --cpus-per-task={cpus_per_task}"
                " --gres=gpu:{n_gpus} {cmd}".format(
                    account_name=self.account_name,
                    partition=self.partition,
                    time=self.time_in_mins,
                    nodes=self.nodes,
                    n_tasks=self.n_tasks,
                    cpus_per_task=2*self.n_gpus,
                    n_gpus=self.n_gpus,
                    cmd=singularity_cmd,
                )
            )
        else:
            full_cmd = "sbatch -A {account_name} -p {partition} -t {time} {cmd}".format(
                account_name=self.account_name,
                partition=self.partition,
                time=self.time_in_mins,
                cmd=singularity_cmd,
            )
        if verbose:
            print(full_cmd)
        call_and_wait(full_cmd, dry=dry)
