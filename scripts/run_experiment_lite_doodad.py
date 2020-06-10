import doodad
print('Using dibya\'s doodad')
try:
    import cloudpickle
except ImportError as e:
    raise ImportError("cloudpickle must be installed inside the docker image")
def failure():
    raise ValueError("Must provide run_method via doodad args!")
fn = doodad.get_args('run_method', failure)
fn()
