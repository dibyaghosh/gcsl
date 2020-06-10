import hashlib
import os
import subprocess
import contextlib
import tempfile

THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.dirname(THIS_FILE_DIR)
EXAMPLES_DIR = os.path.join(REPO_DIR, 'examples')

HASH_BUF_SIZE = 65536 

def hash_file(filename):
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(HASH_BUF_SIZE)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def call_and_wait(cmd, verbose=False, dry=False):
    if dry or verbose:
        print(cmd)
    if not dry:
        p = subprocess.Popen(cmd, shell=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        try:
            print("terminating")
            p.terminate()
        except OSError:
            print("os error!")
            pass
        p.wait()


class CommandBuilder(object):
    def __init__(self):
        self.cmds = []

    def add_command(self, cmd):
        self.cmds.append(cmd)

    def append(self, cmd):
        self.add_command(cmd)

    def extend(self, other):
        if isinstance(other, CommandBuilder):
            self.cmds.extend(other.cmds)
        else:
            self.cmds.extend(other)

    def to_string(self, separator=';'):
        return ';'.join([str(cmd) for cmd in self.cmds])

    def __str__(self):
        return self.to_string()

    def __iter__(self):
        for cmd in self.cmds:
            yield cmd

    def call_and_wait(self, verbose=False, dry=False):
        return call_and_wait(self.to_string())

    @contextlib.contextmanager
    def as_script(self, suffix='.sh'):
        """
        Usage:
        with cmd_builder.as_script() as fname:
            # do stuff with fname
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, mode='w+') as f:
            for cmd in self.cmds:
                f.write(cmd+'\n')
            f.seek(0)
            yield f.name

