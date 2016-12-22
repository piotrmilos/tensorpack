import random

import PIL.Image as Image
from deepsense import neptune
from ..tfutils import get_global_step, get_global_step_var

from tensorpack.callbacks.base import Callback
import tensorflow as tf

# __all__ = ['register_summaries']

NEPTUNE_LOGGER = None

def register_summaries(v, *args):
    if not isinstance(v, list):
        v = [v]
    v.extend(args)
    NEPTUNE_LOGGER.register_summaries(v)

class NeptuneLogger(Callback):
    """
    Sends information to Neptune
    """

    def __init__(self, ctx, logging_step=100):
        global NEPTUNE_LOGGER
        assert NEPTUNE_LOGGER == None, "NeptuneLogger is a singleton class"
        NEPTUNE_LOGGER = self

        self.neptuneContext = ctx
        # This one is automathically mamanged
        self.summariesDict = {}
        # These are channels added by add_channels. They need by be feed manully.
        self.summariesDict2 = {}
        self.logging_step = logging_step
        self.channels_created = False


    def trigger_step(self):
        if not self.channels_created:
            self._create_channels()
            self.channels_created = True

        step = get_global_step()
        if step % self.logging_step == 0 and step>= 100: #We skip some preliminary steps when the averages warm up
            summaryObject = tf.Summary.FromString(self.trainer.summary_op.eval())
            for val in summaryObject.value:
                if val.WhichOneof('value') == 'simple_value' and val.tag in self.summariesDict:
                    channel = self.summariesDict[val.tag]
                    channel.send(x = step, y = val.simple_value)
                # if val.WhichOneof('value') == 'histo' and val.tag in self.summariesDict:
                #     channel = self.summariesDict[val.tag]
                    image = val.image
                    # channel.send(
                    #     x=1,
                    #     y=neptune.Image(
                    #         name='#1 image name',
                    #         description='#1 image description',
                    #         data=Image.open("/home/ubuntu/image1.jpg")))


    def _before_train(self):
        pass
        # This should be invoked here but for some reasons does not work as a work around we invoke it in trigger step
        # self._create_channels()



    def _create_channels(self):
        # We automatically harvest all possibly tags
        summaryObject = tf.Summary.FromString(self.trainer.summary_op.eval())
        for val in summaryObject.value:
            if val.WhichOneof('value') == 'simple_value':
                summaryName = val.tag
                self.summariesDict[summaryName] \
                    = self.neptuneContext.job.create_channel(name=summaryName,
                                                             channel_type=neptune.ChannelType.NUMERIC)
            # if val.WhichOneof('value') == "histo":
            #     summaryName = val.tag + " histogram"
            #     self.summariesDict[summaryName] \
            #         = self.neptuneContext.job.create_channel(name=summaryName,
            #                                                  channel_type=neptune.ChannelType.IMAGE, is_history_persisted=False)

        # Finalize preparation of Neptune
        # self.neptuneContext.job.finalize_preparation()

    def add_channels(self, summariesList):
        for summaryName in summariesList:
            self.summariesDict2[summaryName] \
                = self.neptuneContext.job.create_channel(name=summaryName, channel_type=neptune.ChannelType.NUMERIC)

    def sendMassage(self, channelName, x, y):
        self.summariesDict2[channelName].send(x = x, y = y)

