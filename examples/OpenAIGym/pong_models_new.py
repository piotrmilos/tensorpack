import importlib
import random
from abc import ABCMeta, abstractmethod

import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from examples.OpenAIGym.train_atari_with_neptune import FRAME_HISTORY, DEBUGING_INFO
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.fc import FullyConnected
from tensorpack.models.model_desc import ModelDesc, InputVar, get_current_tower_context
import tensorflow as tf
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.models.pool import MaxPooling

from tensorpack.tfutils import summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.models.nonlin import PReLU
import numpy as np
from tensorpack.callbacks.neptune import NeptuneLogger

from game_processors import _unary_representation


game = "montezuma_revenge"
name = "MontezumaRevengeCodilime-v0"
obs_type = 'image'
nondeterministic = False
frameskip = 1
from gym.envs import registration as rg

rg.register(
    id='{}'.format(name),
    # entry_point='gym.envs.atari:AtariEnv',
    entry_point='examples.OpenAIGym.montezuma_env:MontezumaRevengeFogOfWar',
    kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1},  # A frameskip of 1 means we get every frame
    timestep_limit=frameskip * 100000,
    nondeterministic=nondeterministic,
)

game = "montezuma_revenge"
name = "MontezumaRevengerRandomActionCodilime-v0"
obs_type = 'image'
nondeterministic = False
frameskip = 1

rg.register(
    id='{}'.format(name),
    # entry_point='gym.envs.atari:AtariEnv',
    entry_point='examples.OpenAIGym.montezuma_env:MontezumaRevengeFogOfWarRandomAction',
    kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1},  # A frameskip of 1 means we get every frame
    timestep_limit=frameskip * 100000,
    nondeterministic=nondeterministic,
)

game = "montezuma_revenge"
name = "MontezumaRevengeRandomActionRandomStartingStateCodilime-v0"
obs_type = 'image'
nondeterministic = False
frameskip = 1

rg.register(
    id='{}'.format(name),
    # entry_point='gym.envs.atari:AtariEnv',
    entry_point='examples.OpenAIGym.montezuma_env:MontezumaRevengeFogOfWarRandomActionRandomStartingState',
    kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1},  # A frameskip of 1 means we get every frame
    timestep_limit=frameskip * 100000,
    nondeterministic=nondeterministic,
)
#
# game = "montezuma_revenge"
# name = "MontezumaRevengeFogOfWarResetRestoresImporatantStatesCodilime-v0"
# obs_type = 'image'
# nondeterministic = False
# frameskip = 1
#
# rg.register(
#     id='{}'.format(name),
#     # entry_point='gym.envs.atari:AtariEnv',
#     entry_point='examples.OpenAIGym.montezuma_env:MontezumaRevengeFogOfWarResetRestoresImporatantStates',
#     kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1},  # A frameskip of 1 means we get every frame
#     timestep_limit=frameskip * 100000,
#     nondeterministic=nondeterministic,
# )


class AtariExperimentModel(ModelDesc):
    __metaclass__ = ABCMeta

    def __init__(self):
        raise "The construction should be overriden and define self.input_shape"

    def _get_input_vars(self):
        # assert 1==0, "Aha:{}".format(self.input_shape)
        assert self.number_of_actions is not None
        # print "Number of actions:{}".format(self.number_of_actions)
        inputs = [InputVar(tf.float32, (None,) + self.input_shape, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'futurereward') ]
        return inputs

    # This is a hack due the currect control flow ;)
    def set_number_of_actions(self, num):
        self.number_of_actions = num

    @abstractmethod
    def _get_NN_prediction(self, image):
        """create centre of the graph"""

    @abstractmethod
    def get_screen_processor(self):
        """Get the method used to extract features"""

    @abstractmethod
    def get_history_processor(self):
        """How the history should be processed."""

    def _build_graph(self, inputs):
        state, action, futurereward = inputs
        policy, self.value = self._get_NN_prediction(state)
        self.value = tf.squeeze(self.value, [1], name='pred_value')  # (B,)
        self.logits = tf.nn.softmax(policy, name='logits')

        expf = tf.get_variable('explore_factor', shape=[],
                               initializer=tf.constant_initializer(1), trainable=False)
        logitsT = tf.nn.softmax(policy * expf, name='logitsT')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(self.logits + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, self.number_of_actions), 1)
        advantage = tf.sub(tf.stop_gradient(self.value), futurereward, name='advantage')
        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage, name='policy_loss')
        xentropy_loss = tf.reduce_sum(
            self.logits * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(self.value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(self.value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        summary.add_moving_summary(policy_loss, xentropy_loss, value_loss, pred_reward, advantage)
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        self.cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        self.cost = tf.truediv(self.cost,
                               tf.cast(tf.shape(futurereward)[0], tf.float32),
                               name='cost')

        # print "DEBUGGING INFO:{}".format(DEBUGING_INFO)
        # assert 1 == 0, "AAA"

        if DEBUGING_INFO:
            logits_mean, logits_var = tf.nn.moments(self.logits, axes=[1])
            # logits_mean_r = tf.reduce_sum(logits_mean)
            logits_var_r = tf.reduce_sum(logits_var)
            # tf.scalar_summary('logits_mean', logits_mean_r)
            tf.scalar_summary('logits_var', logits_var_r)


        tf.scalar_summary('entropy beta', entropy_beta)
        tf.scalar_summary('explore factor', expf)

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                SummaryGradient()]

class AtariExperimentFromScreenModel(AtariExperimentModel):

    def __init__(self, parameters):
        self.IMAGE_SIZE = (84, 84)
        self.CHANNELANDHISTORY = 3*FRAME_HISTORY
        print "Frame history:{}".format(FRAME_HISTORY)
        self.input_shape = self.IMAGE_SIZE + (self.CHANNELANDHISTORY,)
        self.image_index = 1
        # print "INPUT_SHAPE:{}".format(self.input_shape)

    def _get_NN_prediction(self, image):
        self._create_unnary_variables_with_summary(image[:, 0, :, 0],
                                                   (10, 10, 6, 6, 6),
                                                   ("rewards", "levels", "lives0", "lives1", "lives2"))
        image = image / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            lc0 = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            lc0 = MaxPooling('pool0', lc0, 2)
            lc1 = Conv2D('conv1', lc0, out_channel=32, kernel_shape=5)
            lc1 = MaxPooling('pool1', lc1, 2)
            lc2 = Conv2D('conv2', lc1, out_channel=64, kernel_shape=4)
            lc2 = MaxPooling('pool2', lc2, 2)
            lc3 = Conv2D('conv3', lc2, out_channel=64, kernel_shape=3)

        lfc0 = FullyConnected('fc0', lc3, 512, nl=tf.identity)
        lfc0 = PReLU('prelu', lfc0)
        policy = FullyConnected('fc-pi', lfc0, out_dim=self.number_of_actions, nl=tf.identity)
        value = FullyConnected('fc-v', lfc0, 1, nl=tf.identity)

        
        # if DEBUGING_INFO:
        #     summary.add_activation_summary(lc0, "conv_0")
        #     summary.add_activation_summary(lc1, "conv_1")
        #     summary.add_activation_summary(lc2, "conv_2")
        #     summary.add_activation_summary(lc3, "conv_3")
        #     summary.add_activation_summary(lfc0, "fc0")
        #     summary.add_activation_summary(policy, "policy")
        #     summary.add_activation_summary(value, "fc-v")

        return policy, value


    def _create_unnary_variables_with_summary(self, vec, sizes, names):
        index = 0
        for size, name in zip(sizes, names):
            for s in xrange(size):
                var_name = name + "_{}".format(s)
                # print "Creating: {}".format(var_name)
                v = tf.slice(vec, [0, index], [-1, 1], name = var_name)
                var_to_vis = tf.reduce_mean(v, name = var_name+"_vis")
                # print "Creating:{}".format(var_name+"_vis")
                summary.add_moving_summary(var_to_vis)
                index += 1

    def _screen_processor(self, image):
        # print "Image:{}".format(image)
        # NeptuneLogger.get_instancetance()
        additonal_data = image[0, 0:84, 0].tolist()
        # worker_id = additonal_data.pop(0)
        # is_imporant_event = additonal_data.pop(0)
        # if is_imporant_event==1:
        #     self.image_index += 1
        #     # print "Impoartant event!"
        #     screenshot_with_debugs_img = self.screenshot_with_debugs(image, worker_id,
        #                                                     additonal_data,
        #                                                     ["rewards", "levels", "lives0", "lives1", "lives2",
        #                                                      "level0", "level1", "level2", "level3", "level4"])
        #     screenshot_with_debugs_img.save("/tmp/screenshot{}.png".format(self.image_index), "PNG")
        #     # NeptuneLogger.get_instance().send_image(screenshot_with_debugs_img)
        #     # self.neptune_image_logger.send_image(screenshot_with_debugs_img)


        additonal_data = _unary_representation(additonal_data, (10, 10, 6, 6, 6))
        additonal_data = np.lib.pad(additonal_data, (0, 84 - additonal_data.size), 'constant', constant_values=(0,0))
        image_new =  cv2.resize(image, self.IMAGE_SIZE[::-1])
        image_new[0,:,0] = additonal_data

        return image_new

    def get_history_processor(self):
        return None


    def get_screen_processor(self):
        return lambda image: self._screen_processor(image)


class AtariExperimentFromScreenStagesResetModel(AtariExperimentFromScreenModel):

    def _get_NN_prediction(self, image):
        self._create_unnary_variables_with_summary(image[:, 0, :, 0],
                                                   (10, 10, 6, 6, 6),
                                                   ("rewards", "levels", "lives0", "lives1", "lives2"))
        NUMBER_OF_REWARD_EVENTS = 10

        rewards_events = []
        for x in xrange(NUMBER_OF_REWARD_EVENTS):
            rewards_events.append(tf.reshape(image[:, 0, x, 0], (-1, 1)))


        image = image / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            lc0 = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            lc0 = MaxPooling('pool0', lc0, 2)
            lc1 = Conv2D('conv1', lc0, out_channel=32, kernel_shape=5)
            lc1 = MaxPooling('pool1', lc1, 2)
            lc2 = Conv2D('conv2', lc1, out_channel=64, kernel_shape=4)
            lc2 = MaxPooling('pool2', lc2, 2)
            lc3 = Conv2D('conv3', lc2, out_channel=64, kernel_shape=3)

        policies = []
        values = []
        for x in xrange(10):
            lfc0 = FullyConnected('fc0{}'.format(x), lc3, 512, nl=tf.identity)
            lfc0 = PReLU('prelu{}'.format(x), lfc0)
            policy = FullyConnected('fc-pi{}'.format(x), lfc0, out_dim=self.number_of_actions, nl=tf.identity)
            value = FullyConnected('fc-v{}'.format(x), lfc0, 1, nl=tf.identity)

            policies.append(policy)
            values.append(value)

        weighted_policies = []
        weighted_values = []

        for weight, policy, value in zip(rewards_events, policies, values):
            weighted_policies.append(tf.multiply(weight, policy))
            weighted_values.append(tf.multiply(weight, value))

        policy = tf.add_n(weighted_policies)
        value = tf.add_n(weighted_values)
        # if DEBUGING_INFO:
        #     summary.add_activation_summary(lc0, "conv_0")
        #     summary.add_activation_summary(lc1, "conv_1")
        #     summary.add_activation_summary(lc2, "conv_2")
        #     summary.add_activation_summary(lc3, "conv_3")
        #     summary.add_activation_summary(lfc0, "fc0")
        #     summary.add_activation_summary(policy, "policy")
        #     summary.add_activation_summary(value, "fc-v")

        return policy, value



class ProcessedAtariExperimentModel(AtariExperimentModel):
    __metaclass__ = ABCMeta

    def __init__(self, parameters):
        params = parameters.split(":")
        self._process_network_parmeters(params[0])
        self._process_input_history_processor(params[1])

    def _process_input_history_processor(self, str):
        params = str.split(" ")
        process_class_name = params[0]

        # module_name, function_name = ctx.params.featureExtractor.split(".")
        module_name = process_class_name[:process_class_name.rfind('.')]
        class_name = process_class_name[process_class_name.rfind('.') + 1:]
        process_class = importlib.import_module(module_name).__dict__[class_name]

        self.data_processor = process_class(params[1:])
        self.input_shape = self.data_processor.get_input_shape()

    @abstractmethod
    def _process_network_parmeters(self, str):
        """How to process the networks parameters"""

    def get_screen_processor(self):
        return lambda image: self.data_processor.process_screen(image)

    def get_history_processor(self):
        return lambda image: self.data_processor.process_history(image)

    def _get_NN_prediction(self, image):
        l = image
        for i in xrange(0, self.numberOfLayers):
            l = FullyConnected('fc{}'.format(i), l, self.numberOfNeurons, nl=tf.identity)
            l = PReLU('prelu{}'.format(i), l)
        policy = FullyConnected('fc-pi', l, out_dim=self.number_of_actions, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return policy, value

class ProcessedAtariExperimentModelFCModel(ProcessedAtariExperimentModel):

    def _process_network_parmeters(self, str):
        # print "AAAA{}".format(str)
        params = str.split(" ")
        self.number_of_layers = int(params[0])
        self.number_of_neurons = int(params[1])

    def _get_NN_prediction(self, image):
        l = image
        for i in xrange(0, self.number_of_layers):
            l = FullyConnected('fc{}'.format(i), l, self.number_of_neurons, nl=tf.identity)
            l = PReLU('prelu{}'.format(i), l)
            # summary.add_activation_summary(l, "fc {} relu output".format(i))
        policy = FullyConnected('fc-pi', l, out_dim=self.number_of_actions, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return policy, value

class ProcessedAtariExperimentModelFCStagedModel(ProcessedAtariExperimentModel):


    def __init__(self, parameters, stage):
        super(self.__class__, self).__init__(parameters)
        self.stage = stage

    def _process_network_parmeters(self, str):
        # Paramters are (number_of_layers, number_of_neurons, extractor_method) or
        # (number_of_layers_0, number_of_neurons_0, number_of_layers_1, number_of_neurons_1, ..,
        # number_of_layers_stage_n, number_of_neurons_stage_n, extractor_method)
        layers_params = str.split(" ")

        layers_params = layers_params*20 # This is to handle the first type of parametes and backward comaptibility

        self.number_of_layers = map(lambda x: int(x), layers_params[::2])
        self.number_of_neurons = map(lambda x: int(x), layers_params[1::2])
        params = str.split(" ")
        self.numberOfLayers = int(params[0])
        self.numberOfNeurons = int(params[1])

    def add_column(self, previous_column_layers, column_num, trainable = True):
        print "Creating column:{}".format(column_num)
        column_prefix = "-column-"
        # column_num = ""
        # print "Adding column:{}".format(column_num)
        new_column = []
        # We append this as this is input
        new_column.append(previous_column_layers[0])
        for i in xrange(1, self.number_of_layers[self.stage]+1):
            input_neurons = new_column[-1]
            l = FullyConnected('fc-{}{}{}'.format(i, column_prefix, column_num),
                               input_neurons, self.number_of_neurons[self.stage], nl=tf.identity, trainable=trainable)
            l = PReLU('prelu-{}{}{}'.format(i, column_prefix, column_num), l)

            if len(previous_column_layers)>i:
                new_layer = tf.concat(1, [previous_column_layers[i], l])
            else:
                new_layer = l
            new_column.append(new_layer)

        last_hidden_layer = new_column[-1]
        policy = FullyConnected('fc-pi{}{}'.format(column_prefix, column_num),
                                last_hidden_layer, out_dim=self.number_of_actions, nl=tf.identity, trainable=trainable)
        value = FullyConnected('fc-v{}{}'.format(column_prefix, column_num),
                               last_hidden_layer, 1, nl=tf.identity, trainable=trainable)

        visible_layer = tf.concat(1, [policy, value])
        new_column.append(visible_layer)
        return new_column, policy, value


    def _get_NN_prediction(self, image):
        print "Current stage is:{}".format(self.stage)
        column = [image]
        for stage in xrange(self.stage):
            column, _, _ = self.add_column(column, stage, trainable = False)

        #The final tower
        column, policy, value = self.add_column(column, self.stage, trainable=True)
        return policy, value
