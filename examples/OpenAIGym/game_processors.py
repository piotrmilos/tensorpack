import numpy as np
from abc import ABCMeta, abstractmethod
from examples.OpenAIGym.train_atari_with_neptune import FRAME_HISTORY

def _unary_representation(values, sizes, type = 0):
    all_results = []
    for x, size_x in zip(values, sizes):
        x = int(x)
        if type==0: #Fake unary but faster
            # print "Fast type"
            r = np.zeros(size_x)
            r[x] = 1
        else:
            # print "Geunine types:({}, {}).".format(size_x, x)
            if isinstance(x, int):
                x = [x]
            if isinstance(size_x, int):
                size_x = [size_x]
            r = np.ones(x)
            pad_params = map(lambda (v1, v2): (0, v1 - v2), zip(size_x, x))
            r = np.lib.pad(r, pad_params, 'constant', constant_values=(0))

        all_results.append(r.flatten())

    return np.concatenate(all_results)


class GameProcessor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def process_screen(self, input):
        """Get the method to process raw imput"""

    @abstractmethod
    def process_history(self, input):
        """Process history"""

    @abstractmethod
    def get_input_shape(self):
        """Return the shape of input"""

class AbstractPongFeatureGameProcessor(GameProcessor):

    @abstractmethod
    def get_input_shape(self):
        """What is the shape."""
        # return 160*self.number_of_features*FRAME_HISTORY

    @abstractmethod
    def process_history(self, input):
        """How to process history"""


    def process_screen(self, observation):
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
        return res

class BasicUnaryFeatureProcessor(AbstractPongFeatureGameProcessor):

    def __init__(self, parameters):
        self.conversion_typ = int(parameters[0])
        assert self.conversion_typ==0 or self.conversion_typ==1, "Other types are not allowed."

    def get_input_shape(self):
        return (160*6*FRAME_HISTORY,)

    def process_history(self, input):
        res = input.flatten()
        res = 100*res
        # print "Position {}".format(res)

        return _unary_representation(res.tolist(), (160,)*(6*FRAME_HISTORY), self.conversion_typ)


class UnaryRepresentationWithBallGradients(AbstractPongFeatureGameProcessor):

    def __init__(self, parameters):
        self.conversion_typ = int(parameters[0])
        assert self.conversion_typ==0 or self.conversion_typ==1, "Other types are not allowed."

    def get_input_shape(self):
        return (160*6*FRAME_HISTORY + 20*6,)

    def process_history(self, input):
        res1 = input.flatten()
        res1 = 100*res1
        first_features_vect =\
            _unary_representation(res1.tolist(), (160,) * (6 * FRAME_HISTORY), self.conversion_typ)

        ball_y = 100*input[0,0,:]
        ball_x = 100*input[0,1,:]
        shift = 10
        proc = lambda x: int( max(min(shift + x, 20),0))

        diff_y = [proc(ball_y[i+1] - ball_y[i]) for i in xrange(3)]

        diff_x = [proc(ball_x[i+1] - ball_x[i]) for i in xrange(3)]

        # print "Differences:{}".format(diff_y)

        diff_y_vect = _unary_representation(diff_y, (20, )*3, self.conversion_typ)
        diff_x_vect = _unary_representation(diff_x, (20, )*3, self.conversion_typ)

        res = np.concatenate((first_features_vect, diff_y_vect, diff_x_vect))

        return res

class DifferencePongFeatureGameProcessor(AbstractPongFeatureGameProcessor):

    def __init__(self, parameters):
        pass

    def get_input_shape(self):
        return (6*FRAME_HISTORY,)

    def process_history(self, input):
        # input[0,0,:] ball_y
        # input[0,1,:] ball_x
        # input[1,0,:] computer_y
        # input[1,1,:] computer_x
        # input[2,0,:] agent_y
        # input[2,1,:] agent_x
        res = np.copy(input)
        res[2,0,:] = res[2,0,:] - res[0,0,:]
        # print "Res:{}".format(res[:,:,3])

        res = input.flatten()
        return res

class BallDifferencesPongFeature2GameProcessor(AbstractPongFeatureGameProcessor):

    def __init__(self, parameters):
        pass

    def get_input_shape(self):
        return (6*FRAME_HISTORY+2,)

    def process_history(self, input):
        # input[0,0,:] ball_y
        # input[0,1,:] ball_x
        # input[1,0,:] computer_y
        # input[1,1,:] computer_x
        # input[2,0,:] agent_y
        # input[2,1,:] agent_x
        res = np.copy(input)
        res[2,0,:] = res[2,0,:] - res[0,0,:]
        # print "AA:{}, {}, {}, {}, {}".format(input[0,0,0], input[0,0,1], input[0,0,2],
        #                                      input[0,0,3], input[0,0,3] + (input[0,0,3] - input[0,0,2]))

        # Prediction of the position of the ball in the next step.
        ball_in_the_next_step =  (input[0,0,3] + (input[0,0,3] - input[0,0,2]),
                                  input[0,1,3] + (input[0,1,3] - input[0,1,2]))
        res2 = np.array(ball_in_the_next_step)

        res = np.concatenate((input.flatten(), res2))
        return res

class BallDifferencesPongFeatureGameProcessor(AbstractPongFeatureGameProcessor):

    def __init__(self, parameters):
        pass

    def get_input_shape(self):
        return (6*FRAME_HISTORY+6,)

    def process_history(self, input):
        # input[0,0,:] ball_y
        # input[0,1,:] ball_x
        # input[1,0,:] computer_y
        # input[1,1,:] computer_x
        # input[2,0,:] agent_y
        # input[2,1,:] agent_x
        res = np.copy(input)
        res[2,0,:] = res[2,0,:] - res[0,0,:]
        diffs =  (input[0,0,0] - input[0,0,1], input[0,0,1] - input[0,0,2], input[0,0,2] - input[0,0,3],
                  input[0,1,0] - input[0,1,1], input[0,1,1] - input[0,1,2], input[0,1,2] - input[0,1,3])
        res2 = np.array(diffs)

        res = np.concatenate((input.flatten(), res2))
        return res

class BasicPongFeatureGameProcessor(AbstractPongFeatureGameProcessor):
    """Do nothing ;)"""
    def __init__(self, parameters):
        pass

    def get_input_shape(self):
        return (6*FRAME_HISTORY,)

    def process_history(self, input):
        res = input.flatten()
        return res