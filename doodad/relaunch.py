"""
Support for relaunching a run from a checkpoint
if a run terminates prematurely.

Notes to self for docker checkpoints:

- Need to enable experimental mode on docker (run ENABLE_CRIU_CMD)
- The docker process cannot be in interactive mode or use tty
- Checkpointing will stop the docker image and save a bunch of files to disk. You need to restart it to keep going.

"""
import uuid

INSTALL_CRIU_CMD = 'apt-get install -y criu'
ENABLE_CRIU_CMD = 'echo "{\"experimental\": true}" >> /etc/docker/daemon.json; systemctl restart docker'

def checkpoint_cmd(docker_name, chk_name, chk_dir='/docker_checkpoints'):
    return 'docker checkpoint create --checkpoint-dir=%s %s %s' % (chk_dir, docker_name, chk_name)

def checkpoint_restore_cmd(docker_name, checkpoint_name, chk_dir='/docker_checkpoints'):
    return 'docker start --checkpoint-dir=%s --checkpoint=%s %s' % (chk_dir, checkpoint_name, docker_name)


class CheckpointManager(object):
    def __init__(self, restore=False, checkpoint_dir='/docker_checkpoints'):
        self.checkpoint_name = uuid.uuid4()
        self.checkpoint_dir = checkpoint_dir
        self.restore = restore

    def checkpoint_and_tar_cmd(self, docker_name, tar_name, restart=True):
        cmds = []
        checkpoint_name = self.checkpoint_name
        chk_cmd = checkpoint_cmd(docker_name, checkpoint_name, chk_dir=self.checkpoint_dir)
        cmds.append(chk_cmd)

        chk_dir = os.path.join(self.checkpoint_dir, checkpoint_name)
        tar_cmd = 'tar -cvf %s %s ' % (tar_name, chk_dir)
        cmds.append(tar_cmd)

        if restart:
            restart_cmd = checkpoint_restore_cmd(docker_name, checkpoint_name, chk_dir=self.checkpoint_dir)
            cmds.append(restart_cmd)
        return ';'.join(cmds)

    def checkpoint_tar_loop_cmd(self, docker_name, tar_name, wait_interval=1):
        chk_tar_cmd = self.checkpoint_and_tar_cmd(docker_name, tar_name)
        """
        while /bin/true; do
            {chk_tar_cmd}
            sleep {wait_interval}
        done & echo sync initiated
        """.format(wait_interval=wait_interval, chk_tar_cmd=chk_tar_cmd,

