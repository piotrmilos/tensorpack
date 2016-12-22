import importlib
import random
from abc import ABCMeta, abstractmethod

import cv2

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
        # print "INPUT_SHAPE:{}".format(self.input_shape)

    def _get_NN_prediction(self, image):
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

    def get_screen_processor(self):
        return lambda image: cv2.resize(image, self.IMAGE_SIZE[::-1])

    def get_history_processor(self):
        return None

# class AtariExperimentFromScreenStagesResetModel(AtariExperimentFromScreenModel):
#
#     def _get_NN_prediction(self, image):
#         rewards_events_0 = image[:, 0, 0, 0]
#         rewards_events_1 = image[:, 0, 0, 1]
#         rewards_events_2 = image[:, 0, 0, 2]
#
#         weight0 = rewards_events_0 * 1 + rewards_events_1 * 0.5 + rewards_events_2 * 0.25
#         weight1 = rewards_events_0 * 0 + rewards_events_1 * 0.5 + rewards_events_2 * 0.25
#         weight2 = rewards_events_0 * 0 + rewards_events_1 * 0.0 + rewards_events_2 * 0.25
#
#
#         m_weight0 = tf.reduce_mean(weight0, name="m_weight0")
#         m_weight1 = tf.reduce_mean(weight1, name="m_weight1")
#         m_weight2 = tf.reduce_mean(weight2, name="m_weight2")
#
#         summary.add_moving_summary(m_weight0)
#         summary.add_moving_summary(m_weight1)
#         summary.add_moving_summary(m_weight2)
#         # tf.scalar_summary('weight0', weight0)
#         # tf.scalar_summary('weight1', weight1)
#         # tf.scalar_summary('weight2', weight2t2)
#
#         weight0 = tf.reshape(weight0, [-1, 1])
#         weight1 = tf.reshape(weight1, [-1, 1])
#         weight2 = tf.reshape(weight2, [-1, 1])
#
#         # version = tf.__version__
#         # assert 1==0, "Version:{}".format(version)
#         #
#         # rewards_events = tf.reshape(rewards_events_0, [-1, 1])
#         # weight0 = rewards_events * 0.5 + (1 - rewards_events) * 1
#         # weight1 = rewards_events * 0.5 + (1 - rewards_events) * 0
#         # weight0 = tf.reshape(weight0, [-1, 1])
#         # weight1 = tf.reshape(weight1, [-1, 1])
#         #
#         # print "Rewards:{} and {} and {}".format(rewards_events, tf.constant(1), tf.Variable(10000))
#         # aaa = tf.equal(rewards_events, tf.constant(0.0))
#         #
#         # weight0 = tf.cond(tf.equal(rewards_events, tf.constant(0.0)), lambda: rewards_events, lambda: rewards_events)
#         # weight1 = tf.cond(tf.equal(rewards_events, tf.constant(0.0)), lambda: tf.constant(0), lambda: tf.Variable(10000))
#         # weight2 = tf.cond(tf.equal(rewards_events, tf.constant(0.0)), lambda: tf.constant(0), lambda: tf.Variable(10000))
#         #
#         # weight0 = tf.cond(tf.equal(rewards_events, tf.constant(1.0)), lambda: tf.constant(0.5), weight0)
#         # weight1 = tf.cond(tf.equal(rewards_events, tf.constant(1.0)), lambda: tf.constant(0.5), weight1)
#         # weight2 = tf.cond(tf.equal(rewards_events, tf.constant(1.0)), lambda: tf.constant(0), weight2)
#         #
#         # weight0 = tf.cond(tf.equal(rewards_events, tf.constant(2.0)), lambda: tf.Variable(0.5, trainable=False), weight0)
#         # weight1 = tf.cond(tf.equal(rewards_events, tf.constant(2.0)), lambda: tf.Variable(0.5, trainable=False), weight1)
#         # weight2 = tf.cond(tf.equal(rewards_events, tf.constant(2.0)), lambda: tf.Variable(0, trainable=False), weight2)
#         #
#         # weight0 = tf.reshape(weight0, [-1, 1], name="weight0")
#         # weight1 = tf.reshape(weight1, [-1, 1], name="weight1")
#         # weight2 = tf.reshape(weight2, [-1, 1], name="weight2")
#         #
#         # tf.scalar_summary('weight0', weight0)
#         # tf.scalar_summary('weight1', weight1)
#         # tf.scalar_summary('weight2', weight2)
#
#
#         # weights = tf.Variable([[1, 0], [0.5, 0.5]], trainable=False)
#         # current_weights = weights[rewards_events,:]
#
#         # print "The number of rewards events are:{}".format(rewards_events)
#
#         image = image / 255.0
#         with argscope(Conv2D, nl=tf.nn.relu):
#             lc0 = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
#             lc0 = MaxPooling('pool0', lc0, 2)
#             lc1 = Conv2D('conv1', lc0, out_channel=32, kernel_shape=5)
#             lc1 = MaxPooling('pool1', lc1, 2)
#             lc2 = Conv2D('conv2', lc1, out_channel=64, kernel_shape=4)
#             lc2 = MaxPooling('pool2', lc2, 2)
#             lc3 = Conv2D('conv3', lc2, out_channel=64, kernel_shape=3)
#
#         lfc00 = FullyConnected('fc00', lc3, 512, nl=tf.identity)
#         lfc00 = PReLU('prelu0', lfc00)
#         policy0 = FullyConnected('fc-pi0', lfc00, out_dim=self.number_of_actions, nl=tf.identity)
#         value0 = FullyConnected('fc-v0', lfc00, 1, nl=tf.identity)
#
#         lfc01 = FullyConnected('fc01', lc3, 512, nl=tf.identity)
#         lfc01 = PReLU('prelu1', lfc01)
#         policy1 = FullyConnected('fc-pi1', lfc01, out_dim=self.number_of_actions, nl=tf.identity)
#         value1 = FullyConnected('fc-v1', lfc01, 1, nl=tf.identity)
#
#         lfc02 = FullyConnected('fc02', lc3, 512, nl=tf.identity)
#         lfc02 = PReLU('prelu2', lfc02)
#         policy2 = FullyConnected('fc-pi2', lfc02, out_dim=self.number_of_actions, nl=tf.identity)
#         value2 = FullyConnected('fc-v2', lfc02, 1, nl=tf.identity)
#
#         policy = tf.add_n([tf.multiply(weight0, policy0), tf.multiply(weight1, policy1), tf.multiply(weight2, policy2)])
#         value = tf.add_n([tf.multiply(weight0, value0), tf.multiply(weight1, value1), tf.multiply(weight2, value2)])
#
#         # if DEBUGING_INFO:
#         #     summary.add_activation_summary(lc0, "conv_0")
#         #     summary.add_activation_summary(lc1, "conv_1")
#         #     summary.add_activation_summary(lc2, "conv_2")
#         #     summary.add_activation_summary(lc3, "conv_3")
#         #     summary.add_activation_summary(lfc0, "fc0")
#         #     summary.add_activation_summary(policy, "policy")
#         #     summary.add_activation_summary(value, "fc-v")
#
#         return policy, value
#
#     def _screen_processor(self, image):
#         # print "Image:{}".format(image)
#         additonal_data = image[0,0,:]
#         image_new =  cv2.resize(image, self.IMAGE_SIZE[::-1])
#         image_new[0,0,:] = additonal_data
#
#         return image_new
#
#     def get_screen_processor(self):
#         return lambda image: self._screen_processor(image)


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
