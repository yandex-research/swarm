import subprocess

def execute(strCMD):
    print('executing:', strCMD, '...')
    try:
        return subprocess.check_output(strCMD, shell=True, universal_newlines=True)
    except:
        return None


for i in range(6, 7):
    untar = "find . -name 'urlsf_subset{0:02d}*.xz' -execdir tar -xvf '{1}' \;".format(i, '{}')
    shard = "find . -type f -name '*-*.txt' -exec cat {1} + >> shards/shard{0}.shard".format(i, '{}')

    for i in range(10):
        delete = "rm -rf {0:02d}*.txt".format(i)

        execute(delete)
    delete = 'rm -rf *txt'
    execute(delete)

    execute(untar)
    execute(shard)
