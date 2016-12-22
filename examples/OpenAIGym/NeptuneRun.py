import sys
import os


reload(sys)
# We do support UTF-8 encoding only.
sys.setdefaultencoding('UTF8')  # For current process.
os.environ['PYTHONIOENCODING'] = 'UTF-8'  # For child processes.
from deepsense.neptune.cli.run import run

import pip
installed_packages = pip.get_installed_distributions()

HOME_DIR = os.getenv("HOME")
HOME_DIR = "/home/piotr.milos"
ROOT = os.path.join(HOME_DIR, "rl2")

# ROOT = os.path.join(HOME_DIR, "PycharmProjects/rl2")

EXPERIMENTS_DIR = os.path.join(ROOT, "NeptuneIntegration", "Experiments")
EXPERIMENTS_DUMPS = os.path.join(ROOT, "NeptuneIntegration", "Neptune_dumps")

if len(sys.argv)==1:
    print "Provide the name of the experiment (the name of the configuration yaml file without suffix"
    sys.exit(1)

experiment_name = sys.argv[1]
experiment_yaml = os.path.join(EXPERIMENTS_DIR, experiment_name+".yaml")
experiment_dump = os.path.join(EXPERIMENTS_DUMPS, experiment_name)

paramtersList = ["run", "train_atari_with_neptune_proxy.py", "--config", experiment_yaml, "--dump-dir", experiment_dump,
                 "--tags", "pmilos", "rl", "pong"]

run(paramtersList)