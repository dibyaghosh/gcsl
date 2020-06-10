import os

class SSHCredentials(object):
    """
    Container for SSH credentials

    Args:
        hostname (str):
        username (str):
        password (str, optional):
            Authenticate via plain-text password. This features requires the 'sshpass' program to be installed.
            This usage is not suggested due to security reasons.
        identity_file (str, optional):
            Path to a private key file for SSL public key authentication
    """
    def __init__(self, hostname=None, username=None, password=None, identity_file=None):
        assert password is not None or identity_file is not None, "One of password or identity_file must be provided"
        self.hostname = hostname
        self.username = username
        self.password = password
        self.identity_file = os.path.expanduser(identity_file)

    def get_ssh_cmd_prefix(self):
        """
        Return a command prefix
            Ex.
            'ssh user@host -i id_file '
        """
        cmd = 'ssh %s@%s' % (self.username, self.hostname)
        if self.identity_file:
            cmd += ' -i %s' % self.identity_file
        elif self.password:
            cmd = 'sshpass -p \'%s\' %s' % (self.password, cmd)
            print('WARNING: Using password-based ssh is not secure! Please consider using identity files.')
        else:
            raise NotImplementedError()
        return cmd + ' '

    def get_ssh_bash_cmd(self, cmd):
        prefix = self.get_ssh_cmd_prefix()
        return prefix + " '%s'"%cmd

    def get_ssh_script_cmd(self, script_name):
        cmd = 'ssh %s@%s' % (self.username, self.hostname)
        if self.identity_file:
            cmd += ' -i %s' % self.identity_file
        else:
            raise NotImplementedError()
        cmd += " 'bash -s' < %s" % script_name
        return cmd

    def get_scp_cmd(self, source, destination, recursive=True):
        cmd = 'scp'
        if recursive:
            cmd += ' -r'
        if self.identity_file:
            cmd += ' -i %s' % self.identity_file
        else:
            raise NotImplementedError()
        cmd += ' %s' % source
        cmd += ' %s@%s:%s' % (self.username, self.hostname, destination)
        return cmd

    @property
    def user_host(self):
        return '%s@%s' % (self.username, self.hostname)
