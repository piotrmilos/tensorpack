import importlib
import random
from abc import ABCMeta, abstractmethod

import cv2

from examples.OpenAIGym.train_atari_with_neptune import FRAME_HISTORY
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


class AtariExperimentModel(ModelDesc):
    __metaclass__ = ABCMeta

    def __init__(self):
        raise "The construction should be overriden and define self.input_shape"

    def _get_input_vars(self):
        assert self.number_of_actions is not None
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

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                SummaryGradient()]

class AtariExperimentFromScreenModel(AtariExperimentModel):

    def __init__(self, parameters):
        self.IMAGE_SIZE = (84, 84)
        self.CHANNELANDHISTORY = 3*4
        self.input_shape = self.IMAGE_SIZE + (self.CHANNELANDHISTORY,)

    def _get_NN_prediction(self, image):
        image = image / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        policy = FullyConnected('fc-pi', l, out_dim=self.number_of_actions, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)

        return policy, value

    def get_screen_processor(self):
        return lambda image: cv2.resize(image, self.IMAGE_SIZE[::-1])

    def get_history_processor(self):
        return None



class ProcessedPongExperiment(AtariExperimentModel):

    def __init__(self, parameters):
        params = parameters.split(" ")
        self.numberOfLayers = int(params[0])
        self.numberOfNeurons = int(params[1])
        process_class_name = params[2]
        process_class = globals()[process_class_name]
        self.data_processor = process_class(params[3:])
        self.input_shape = self.data_processor.get_input_shape()

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


class SimplifiedPongExperimentModel(AtariExperimentModel):
    __metaclass__ = ABCMeta

    def __init__(self, parameters):
        self.number_of_layers = int(parameters.split(" ")[0])
        self.number_of_neurons = int(parameters.split(" ")[1])
        self.extractor_method_name = parameters.split(" ")[2]
        self.extractor_method =  getattr(SimplifiedPongExperimentModel, self.extractor_method_name)
        assert hasattr(SimplifiedPongExperimentModel, self.extractor_method_name + "_input_shape"), "An extractor " \
                                                "method xyz must be accompanied by a variable xyz_input_shape," \
                                                "who secify the shape of the network input"
        self.input_shape = getattr(SimplifiedPongExperimentModel, self.extractor_method_name + "_input_shape")


    def get_screen_processor(self):
        return self.extractor_method

    def get_history_processor(self):
        return None

    # The shape of the results is forced by the rest of the code (e.g. the history handler)
    # These functions should not be altered unless a bug if found.
    # If you want a new functaionality just write a new one.
    findPongObjectsNumpyDifference_input_shape = (3, 2) + (FRAME_HISTORY,)

    @staticmethod
    def findPongObjectsNumpyDifference(observation):
        # print "AA"
        ball = (0.84, 0.6)
        computerPlayer = (0, 0)
        agentPlayer = (0, 0)

        positons = np.argwhere(observation[34:193, 15:145, 0] == 236)
        if positons.size != 0:
            ball = [a / 100.0 for a in positons[0]]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 213)
        if positons.size != 0:
            computerPlayer = [a / 100.0 for a in positons[0]]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 92)
        if positons.size != 0:
            agentPlayer = [a / 100.0 for a in positons[0]]

        v1 = [[ball[0]], [ball[1]]]
        v2 = [[computerPlayer[0]], [agentPlayer[0]]]
        v3 = [[computerPlayer[0] - ball[0]], [agentPlayer[0] - ball[0]]]
        res = np.asanyarray((v1, v2, v3))
        # print "Game status:{}".format(res.reshape((6,)))
        # res = np.asarray(([[1], [2]], [[3], [4]], [[5],[6]]))
        return res

    findPongObjectsNumpyStandard_input_shape = (3, 2) + (FRAME_HISTORY,)
    @staticmethod
    def findPongObjectsNumpyStandard(observation):
        ball = ([0.84], [0.6])
        computerPlayer = ([0.11], [0.12])
        agentPlayer = ([0.22], [0.23])

        positons = np.argwhere(observation[34:193, 15:145, 0] == 236)
        if positons.size != 0:
            ball = [[a / 100.0] for a in positons[0]]
            # ballPerturbed = ([ball[0][0] + random.uniform(-0.02, 0.02)], [ball[1][0] - random.uniform(-0.02, 0.02)])

        positons = np.argwhere(observation[34:193, 15:145, 0] == 213)
        if positons.size != 0:
            computerPlayer = [[a / 100.0] for a in positons[0]]
            # computerPlayerPertrubed = ([computerPlayer[0][0] + random.uniform(-0.02, 0.02)], [computerPlayer[1][0] - random.uniform(-0.02, 0.02)])

        positons = np.argwhere(observation[34:193, 15:145, 0] == 92)
        if positons.size != 0:
            agentPlayer = [[ a / 100.0] for a in positons[0]]
            # agentPlayerPertrubed = ([agentPlayer[0][0] + random.uniform(-0.02, 0.02)], [agentPlayer[1][0] - random.uniform(-0.02, 0.02)])

        res = np.asanyarray((ball, computerPlayer, agentPlayer))
        # print "X ==================== X\n"
        # print res.reshape((6,))
        # print "Y ==================== Y\n"
        return res

    findPongObjectsNumpyFlat_input_shape = (6 * FRAME_HISTORY,)
    @staticmethod
    def findPongObjectsNumpyFlat(observation):
        ball = (0.84, 0.6)
        computerPlayer = (0, 0)
        agentPlayer = (0, 0)

        positons = np.argwhere(observation[34:193, 15:145, 0] == 236)
        if positons.size != 0:
            ball = [a / 100.0 for a in positons[0]]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 213)
        if positons.size != 0:
            computerPlayer = [a / 100.0 for a in positons[0]]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 92)
        if positons.size != 0:
            agentPlayer = [a / 100.0 for a in positons[0]]

        res = np.asanyarray((ball, computerPlayer, agentPlayer))
        res =  res.reshape((6,))
        # print res
        return res

    # Well it is not quite unary ;)
    @staticmethod
    def _unary_pack(offset, *args):
        assert 1==0, "this method is deprected.  _unary_representation() should be used instead"
        res = np.zeros(offset*len(args))
        current_offset = 0
        for a in args:
            res[a + current_offset] = 1
            current_offset += offset
        return res

    findPongObjectsNumpyExtended_input_shape = (160 * 6 * FRAME_HISTORY,)
    @staticmethod
    def findPongObjectsNumpyExtended(observation):
        ball = (84, 6)
        computerPlayer = (0, 0)
        agentPlayer = (0, 0)

        positons = np.argwhere(observation[34:193, 15:145, 0] == 236)
        if positons.size != 0:
            ball = positons[0]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 213)
        if positons.size != 0:
            computerPlayer = positons[0]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 92)
        if positons.size != 0:
            agentPlayer = positons[0]

        return SimplifiedPongExperimentModel._unary_pack(160, ball[0], ball[1], computerPlayer[0],
                                computerPlayer[1], agentPlayer[0], agentPlayer[1])


    findPongObjectsNumpyDifferenceExtended_input_shape = (160 * 6 * FRAME_HISTORY,)
    @staticmethod
    def findPongObjectsNumpyDifferenceExtended(observation):
        ball = (84, 6)
        computerPlayer = (0, 0)
        agentPlayer = (0, 0)

        positons = np.argwhere(observation[34:193, 15:145, 0] == 236)
        if positons.size != 0:
            ball = positons[0]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 213)
        if positons.size != 0:
            computerPlayer = positons[0]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 92)
        if positons.size != 0:
            agentPlayer = positons[0]

        diff1 = max(ball[0] - agentPlayer[0], 0)
        diff2 = max(agentPlayer[0] - ball[0], 0)
        # print "Informations to send {} {} {} {} {} {}".format(ball[0], ball[1], computerPlayer[0], agentPlayer[0], diff1, diff2)

        return _unary_representation((ball[0], ball[1], computerPlayer[0], agentPlayer[0], diff1, diff2), (160,)*6)

    @staticmethod
    def _create_unary_matrix(x, y, size):
        ones = np.ones((x, y))
        return np.lib.pad(ones, ((0, size-x), (0, size -y)), 'constant', constant_values=(0))


    DOWNSAMPLE = 1
    INPUT_SIZE = 160 / DOWNSAMPLE
    findPongObjectsFullUnary_input_shape = (INPUT_SIZE * INPUT_SIZE * 2 * FRAME_HISTORY,)
    @staticmethod
    def findPongObjectsFullUnary(observation):
        # print "True unnary!"
        ball = (84, 6)
        computerPlayer = (1, 1)
        agentPlayer = (1, 1)

        positons = np.argwhere(observation[34:193, 15:145, 0] == 236)
        if positons.size != 0:
            ball = positons[0]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 213)
        if positons.size != 0:
            computerPlayer = positons[0]

        positons = np.argwhere(observation[34:193, 15:145, 0] == 92)
        if positons.size != 0:
            agentPlayer = positons[0]


        #
        # ball_array = SimplifiedPongExperimentModel.\
        #     _create_unary_matrix(ball[0] / SimplifiedPongExperimentModel.DOWNSAMPLE,
        #                          balogitsll[1] / SimplifiedPongExperimentModel.DOWNSAMPLE,
        #                          SimplifiedPongExperimentModel.INPUT_SIZE)
        # agents_array = SimplifiedPongExperimentModel.\
        #     _create_unary_matrix(computerPlayer[0] / SimplifiedPongExperimentModel.DOWNSAMPLE,
        #                          agentPlayer[0] / SimplifiedPongExperimentModel.DOWNSAMPLE,
        #                         SimplifiedPongExperimentModel.INPUT_SIZE)
        #

        _size = SimplifiedPongExperimentModel.INPUT_SIZE
        _ds = SimplifiedPongExperimentModel.DOWNSAMPLE
        ball_array = np.zeros((_size, _size))
        agents_array = np.zeros((_size, _size))

        ball_array[ball[0]/_ds, ball[1]/_ds] = 1
        agents_array[computerPlayer[0]/_ds, agentPlayer[0]/_ds] =1

        concatenated = np.concatenate((ball_array, agents_array))

        res = np.reshape(concatenated, (SimplifiedPongExperimentModel.INPUT_SIZE*
                                        SimplifiedPongExperimentModel.INPUT_SIZE*2,))

        return res

class SimplifiedPongExperimentModelFC(SimplifiedPongExperimentModel):
    def _get_NN_prediction(self, image):
        l = image
        for i in xrange(0, self.number_of_layers):
            l = FullyConnected('fc{}'.format(i), l, self.number_of_neurons, nl=tf.identity)
            l = PReLU('prelu{}'.format(i), l)
            # summary.add_activation_summary(l, "fc {} relu output".format(i))
        policy = FullyConnected('fc-pi', l, out_dim=self.number_of_actions, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return policy, value

class SimplifiedPongAtariExperimentModelFCFlatten(SimplifiedPongExperimentModel):
    def _get_NN_prediction(self, image):
        l = tf.reshape(image, [-1, 24])
        for i in xrange(0, self.number_of_layers):
            l = FullyConnected('fc{}'.format(i), l, self.number_of_neurons, nl=tf.identity)
            l = PReLU('prelu{}'.format(i), l)
        policy = FullyConnected('fc-pi', l, out_dim=self.number_of_actions, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return policy, value

class SimplifiedPongExperimentModelFCWithImpactPoint(SimplifiedPongExperimentModel):

    def _get_NN_prediction(self, image):
        l = tf.reshape(image, [-1, 24])
        # This calculates the position of ball when hitting the plane of the agent
        xNew = image[:, 0, 1, 3]
        yNew = image[:, 0, 0, 3]
        xOld = image[:, 0, 1, 2]
        yOld = image[:, 0, 0, 2]
        yPredicted = yNew + (yNew - yOld) * (0.125 - xNew) / (xNew - xOld + 0.005)
        yPredictedTruncated = tf.maximum(tf.minimum(yPredicted, 1), -1)
        yPredictedTruncated = tf.expand_dims(yPredictedTruncated, 1)
        summary.add_activation_summary(yPredictedTruncated, "yPredicted")

        l = tf.concat(1, [l, yPredictedTruncated])

        for i in xrange(0, self.number_of_layers):
            l = FullyConnected('fc{}'.format(i), l, self.number_of_neurons, nl=tf.identity)
            l = PReLU('prelu{}'.format(i), l)
            # summary.add_activation_summary(l, "fc {} relu output".format(i))
        policy = FullyConnected('fc-pi', l, out_dim=self.number_of_actions, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return policy, value


class SimplifiedPongExperimentStagedFC(SimplifiedPongExperimentModel):


    def __init__(self, parameters, stage):
        # Paramters are (number_of_layers, number_of_neurons, extractor_method) or
        # (number_of_layers_0, number_of_neurons_0, number_of_layers_1, number_of_neurons_1, ..,
        # number_of_layers_stage_n, number_of_neurons_stage_n, extractor_method)
        params = parameters.split(" ")
        self.extractor_method_name = params[-1]

        layers_params = params[:-1]
        layers_params = layers_params*20 # This is to handle the first type of parametes and backward comaptibility



        self.number_of_layers = map(lambda x: int(x), layers_params[::2])
        self.number_of_neurons = map(lambda x: int(x), layers_params[1::2])

        # print self.number_of_layers
        # print self.number_of_neurons
        # assert False, "Tak tak"

        self.extractor_method = getattr(SimplifiedPongExperimentModel, self.extractor_method_name)
        assert hasattr(SimplifiedPongExperimentModel, self.extractor_method_name + "_input_shape"), "An extractor " \
                                                                                                    "method xyz must be accompanied by a variable xyz_input_shape," \
                                                                                                    "who secify the shape of the network input"
        self.input_shape = getattr(SimplifiedPongExperimentModel, self.extractor_method_name + "_input_shape")
        self.stage = stage


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

class NonMarkovExperimentModel(AtariExperimentModel):

    def __init__(self, parameters):
        self.input_shape = (30, 30*FRAME_HISTORY)
        self.numberOfLayers = int(parameters.split(" ")[0])
        self.numberOfNeurons = int(parameters.split(" ")[1])

    def get_screen_processor(self):
        return lambda x:x

    def _get_NN_prediction(self, image):

        l = image
        for i in xrange(0, self.numberOfLayers):
            l = FullyConnected('fc{}'.format(i), l, self.numberOfNeurons, nl=tf.identity)
            l = PReLU('prelu{}'.format(i), l)
            # summary.add_activation_summary(l, "fc {} relu output".format(i))
        policy = FullyConnected('fc-pi', l, out_dim=self.number_of_actions, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return policy, value