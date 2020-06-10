import subprocess

def s3_exists(bucket, path, region=None):
    cmd = 'aws s3 ls s3://%s/%s' % (bucket, path)
    if region is not None:
        cmd += ' --region %s'%region
    try:
        result = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        return False

    if result:
        return True
    else:
        return False


def s3_upload(local_file_name, s3_bucket, s3_path, dry=False, region=None):
    remote_path = "s3://%s/%s" % (s3_bucket, s3_path)
    if region is None:
        upload_cmd = ["aws", "s3", "cp", local_file_name, remote_path]
    else:
        upload_cmd = ["aws", "s3", "cp", '--region', region, local_file_name, remote_path]
    print(' '.join(upload_cmd))
    if not dry:
        subprocess.check_call(upload_cmd)
    return remote_path


if __name__ == "__main__":
    print(s3_exists('rail.ex2.a3c', 'jello/mount/tmp04cpn0zj.tar'))
    print(s3_exists('rail.ex2.a3c', 'jello/mount/tmp04cpn0zj2.tar'))

