# The file which is the neptune run target is copied internally and thus cannot be debugged remotely.
#  Making this proxy enables debugging other files.
# Moreover we read parameters here

import sys
import os.path as osp

this_dir = osp.dirname(__file__)
print "This dir:{}".format(this_dir)

examples_dir = osp.join(this_dir, "..")
tensor_pack_dir = osp.join(this_dir, "..", "..")

sys.path.insert(0, this_dir)
# sys.path.insert(0, examples_dir)
sys.path.insert(0, tensor_pack_dir)

print "System paths:{}".format(sys.path)

from deepsense import neptune

from examples.OpenAIGym.train_atari_with_neptune import run_training

ctx = neptune.Context()

run_training(ctx)
