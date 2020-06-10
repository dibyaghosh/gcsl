import os
import configparser
import io

class AWSCredentials(object):
    """
    Container for AWS credential information

    The from_env or from_config option can be used to avoid having key information inside source code

    Args:
        aws_key (str): AWS key
        aws_secret (str): AWS secret key
        from_env (bool): If True, reads key and secret from environment variables
        env_key (str, optional): Environment variable for AWS key. Default AWS_ACCESS_KEY.
        env_secret_key (str, optional): Environment variable for AWS secret key. Default AWS_ACCESS_SECRET.
        from_config (bool): If True, reads key from config file
        config_filename (str, optional): 
    """
    def __init__(self, aws_key=None, aws_secret=None, 
            from_env=False, 
            from_config=False,
            config_filename='~/.aws/credentials',
            env_secret_key='AWS_ACCESS_SECRET',
            env_key='AWS_ACCESS_KEY'):
        self.key = aws_key
        self.secret = aws_secret
        self.from_env=from_env
        if from_env:
            self.key = os.environ.get(env_key)
            self.secret = os.environ.get(env_secret_key)
        if from_config:
            with open(os.path.expanduser(config_filename)) as f:
                sample_config = f.read()
            config = configparser.RawConfigParser(allow_no_value=True)
            config.read_string(sample_config)
            self.key = config.get('default', 'aws_access_key_id')
            self.secret = config.get('default', 'aws_secret_access_key')

    @property
    def aws_key(self):
        return self.key

    @property
    def aws_secret_key(self):
        return self.secret
