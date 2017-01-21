import glob
import os
import random
import threading

import PIL.Image as Image
from deepsense import neptune
from ..tfutils import get_global_step, get_global_step_var

from tensorpack.callbacks.base import Callback
import tensorflow as tf

# __all__ = ['register_summaries']

NEPTUNE_LOGGER = None


class NeptuneLogger(Callback):

    """
    Sends information to Neptune
    """

    INSTANCE = None

    @staticmethod
    def get_instance():
        if NeptuneLogger.INSTANCE == None:
            print "Creating NeptuneLogger instance"
            ctx = neptune.Context()
            NeptuneLogger.INSTANCE = NeptuneLogger(ctx)

        # and channels: {}
        # ".format(id(self), self.image_channel
        #
        # obj = NeptuneLogger.INSTANCE
        # print "Getting instance:{} of type:{} and channels: {} with getter:{} in " \
        #       "thread: {}, pid: {}".format(id(obj), type(obj), obj.image_channel, obj.get_image_channel(), threading.current_thread, os.getpid())

        return NeptuneLogger.INSTANCE


    # def __copy__(self):
    #     raise AttributeError
    #
    # def __deepcopy__(self):
    #     raise AttributeError

    def get_image_channel(self):
        return self.image_channel

    def __init__(self, ctx, logging_step=100):
        global NEPTUNE_LOGGER

        NeptuneLogger.INSTANCE = self
        assert NEPTUNE_LOGGER == None, "NeptuneLogger is a singleton class"
        NEPTUNE_LOGGER = self

        self.neptuneContext = ctx
        # This one is automathically mamanged
        self.summariesDict = {}
        # These are channels added by add_channels. They need by be feed manully.
        self.summariesDict2 = {}
        self.logging_step = logging_step
        self.channels_created = False
        self.image_channel = None
        self.image_index = 0
        self._id = random.randint(1, 100)
        self.image_index = 0


    def trigger_step(self):
        obj = self
        # print "Neptune instance:{} of type:{} and channels: {} with getter:{} in " \
        #       "thread: {}, pid: {}".format(id(obj), type(obj), obj.image_channel, obj.get_image_channel(), threading.current_thread, os.getpid())

        if not self.channels_created:
            self._create_channels()
            self._create_charts((10, 10, 6, 6, 6), ("rewards", "levels", "lives0", "lives1", "lives2"))
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

        images_list = glob.glob("/tmp/screen*.png")
        for f in images_list:
            try:
                self.image_index += 1
                pil_image = Image.open(f)
                self.image_channel.send(x=self.image_index,
                                        y=neptune.Image(
                                        name='screenshot',
                                        description=self.image_index,
                                        data=pil_image))
                os.remove(f)
            except IOError:
                print "Something went wrong with:{}".format(f)
        # pil_image = Image.open("/tmp/screen1000.png")

#


    def _before_train(self):
        pass
        # This should be invoked here but for some reasons does not work as a work around we invoke it in trigger step
        # self._create_channels()



    def _create_channels(self):

        # print "Creating channels for {}".format(self._id)

        self.image_channel = self.neptuneContext.\
            job.create_channel(name="game screenshots", channel_type=neptune.ChannelType.IMAGE)
        # print "Creating channels: {}!".format(self.image_channel)
        # We automatically harvest all possibly tags
        summaryObject = tf.Summary.FromString(self.trainer.summary_op.eval())
        for val in summaryObject.value:
            if val.WhichOneof('value') == 'simple_value':
                summaryName = val.tag
                # print "Summary name:{}".format(summaryName)
                self.summariesDict[summaryName] \
                    = self.neptuneContext.job.create_channel(name=summaryName,
                                                             channel_type=neptune.ChannelType.NUMERIC)
            # if val.WhichOneof('value') == "histo":
            #     summaryName = val.tag + " histogram"
            #     self.summariesDict[summaryName] \
            #         = self.neptuneContext.job.create_channel(name=summaryName,
            #                                                  channel_type=neptune.ChannelType.IMAGE, is_history_persisted=False)

        # print "ALL channels:{}".format(self.neptuneContext.job._channels)
        # print "self.image_channel:{}".format(self.image_channel)
        # self.xxx = 15
        # self.send_image_1(None)



    def _create_charts(self, sizes, names):
        channel_names = self.summariesDict.keys()


        for size, name in zip(sizes, names):
            series = {}
            for s in xrange(size):
                var_name = name + "_{}".format(s)+"_vis"
                series[var_name] = self.summariesDict[var_name]
                channel_names.remove(var_name)
            self.neptuneContext.job.create_chart(name, series=series)

        for n in channel_names:
            self.neptuneContext.job.create_chart(n, series={n: self.summariesDict[n]})

    def send_image(self, pil_image):
        self.image_queue.append(pil_image)
        # self.image_index += 1
        # if self.image_index > 200 and self.image_index%100 ==0 :
        #         print "Sending image of index:{} to {}".format(self.image_index, pil_image, self.image_channel)
        #         pil_image.save("/tmp/screen{}.png".format(self.image_index), "PNG")
        #         self.image_channel.send(x=self.image_index,
        #                                 y=neptune.Image(
        #                                 name='screenshot',
        #                                 description='debuging screenshot',
        #                                 data=pil_image))
    #
    # def send_image_1(self, pil_image):
    #     print "self.image_channel2:{}".format(self.image_channel)
    #     print "Trying to send a picture:{} and {}".format(self._id, self.image_channel)
    #     print "ALL channels here:{}".format(neptune.Context().job._channels)
    #     print "XXX:{}".format(self.xxx)
    #     self.image_index += 1
    #     # if self.image_index > 200:
    #     #     print "Trying to send a picture:{} and ".format(self._id, self.image_channel)
    #     #     print "ALL channels here:{}".format(neptune.Context().job._channels)
    #     #     # if self.image_channel != None:
    #         #     print "Sending image!"
    #         #     self.image_channel.send(x=self.image_index,
    #         #                             y=neptune.Image(
    #         #                             name='screenshot',
    #         #                             description='debuging screenshot',
    #         #                             data=pil_image))
    #



    def add_channels(self, summariesList):
        for summaryName in summariesList:
            self.summariesDict2[summaryName] \
                = self.neptuneContext.job.create_channel(name=summaryName, channel_type=neptune.ChannelType.NUMERIC)

    def sendMassage(self, channelName, x, y):
        self.summariesDict2[channelName].send(x = x, y = y)

