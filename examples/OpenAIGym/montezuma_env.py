import random

from gym.envs.atari import AtariEnv
import numpy as np
import collections

class MontezumaRevengeFogOfWar(AtariEnv):
    def __init__(self, game='pong', obs_type='ram', frameskip=(2, 5), repeat_action_probability=0.):
        super(MontezumaRevengeFogOfWar, self).__init__(game, obs_type, frameskip, repeat_action_probability)


        self._fogs_of_war = {}
        for x in xrange(10):
            self._fogs_of_war[x] = np.ones((21,16), dtype="uint8")
        self.board_id = 0
        self.upsampling_matrix = 255*np.ones((10, 10), dtype="uint8")

        self.current_number_of_lives = 5

        self._rewards_events = 0
        self._internal_id = random.randint(100,999)
        # print "Creating a Montezuma agent with id:{}".format(self._internal_id)


    def _get_image(self):
        self.ale.getScreenRGB(self._buffer)  # says rgb but actually bgr
        res = self._buffer[:, :, [2, 1, 0]]
        fog_of_war = np.kron(self._fogs_of_war[self.board_id], self.upsampling_matrix)
        res[:,:,2] = fog_of_war
        res[0,0,:] = 0
        # print "REW env:{}".format(self._rewards_events)
        # pos = min(self._rewards_events, 2)

        # res[0, 0, self._rewards_events] = 1

        return res

    # def _locate_agent(self):
    #     HACK_PARAMETER = 30
    #     positions = np.argwhere(self._buffer[HACK_PARAMETER:,:,2]==200)
    #
    #     if len(positions) != 0:
    #         return (positions[0,0]+ HACK_PARAMETER, positions[0,1])
    #     # pos = np.argwhere((self._buffer[:, :, 2] == 200)
    #     #                         & (self._buffer[:, :, 1] == 72)
    #     #                         & (self._buffer[:, :, 0] == 72))
    #     #
    #     # counter = collections.Counter(pos[:,1].tolist())
    #     # for x in counter:
    #     #     if counter[x]==10:
    #     #         break
    #     #
    #     # pos_pos = np.argwhere(pos[:,1] == x)
    #     #
    #     # if pos_pos.size != 0:
    #     #     return pos[pos_pos[0]][0]
    #
    #
    # # def set_reward_vector(self, vec):
    # #     self.reward_
    #
    # def _step(self, a):
    #     FOG_OF_WAR_REWARD = 1
    #     TRUE_REWARD_FACTOR = 5
    #     DEATH_REWARD = - 100
    #
    #
    #
    #     ob, reward, game_over, d = super(MontezumaRevengeFogOfWar, self)._step(a)
    #     reward = TRUE_REWARD_FACTOR * reward
    #
    #
    #     if reward !=0:
    #         self._fogs_of_war[self.board_id] = np.ones((21,16), dtype="uint8")
    #         self._rewards_events += 1
    #
    #     self._set_board(ob)
    #
    #     if self._number_of_lives(ob) != self.current_number_of_lives:
    #         self.current_number_of_lives = self._number_of_lives(ob)
    #         reward += DEATH_REWARD
    #
    #
    #     loc = self._locate_agent()
    #     if loc != None:
    #         # print "Agent location:{}".format(loc)
    #         l_x = loc[0]/10
    #         l_y = loc[1]/10
    #         if self._fogs_of_war[self.board_id][l_x, l_y] == 1:
    #             reward += FOG_OF_WAR_REWARD
    #             self._fogs_of_war[self.board_id][l_x, l_y] = 0
    #
    #     return ob, reward, game_over, d
    #
    #
    # def _set_board(self, ob):
    #     if np.all(ob[94:133, 159, 1] == 158):
    #         self.board_id = 0
    #     else:
    #         self.board_id = 1
    #     # print "Board id:{}".format(self.board_id)
    #
    # def _reset(self):
    #     for x in xrange(10):
    #         self._fogs_of_war[x] = np.ones((21,16), dtype="uint8")
    #     self.current_number_of_lives = 5
    #     self._rewards_events = 0
    #     return super(MontezumaRevengeFogOfWar, self)._reset()
    #
    #
    # def _number_of_lives(self, observation):
    #     return int(np.sum(observation[15, :, 0])) / (3 * 210)


class MontezumaRevengeFogOfWarRandomAction(MontezumaRevengeFogOfWar):

    def _step(self, a):
        # print "{}".format(self._internal_id)
        action = a
        if random.uniform(0,1) < 0.05:
            action = random.randint(0,17)

        return super(MontezumaRevengeFogOfWarRandomAction, self)._step(action)

#
# class MontezumaRevengeFogOfWarResetRestoresImporatantStates(MontezumaRevengeFogOfWar):
#
#     def __init__(self, game='pong', obs_type='ram', frameskip=(2, 5), repeat_action_probability=0.):
#         super(MontezumaRevengeFogOfWarResetRestoresImporatantStates, self).__init__(game, obs_type, frameskip, repeat_action_probability)
#
#         self.state_checkpoints = []
#         self.total_true_reward = 0
#
#         self.state_checkpoints.append(self.createCheckPoint())
#
#
#     def createCheckPoint(self):
#         return (self.ale.cloneState(), self.current_number_of_lives,
#                 self.total_true_reward, self.board_id, self._fogs_of_war)
#
#     def restoreCheckPoint(self, chkp):
#         state, current_number_of_lives, total_true_reward, board_id, fogs = chkp
#         self.ale.restoreState(state)
#         self.current_number_of_lives = current_number_of_lives
#         self.total_true_reward = total_true_reward
#         self.board_id = board_id
#         self._fogs_of_war = np.copy(fogs)
#         print "Restoring no of lives:{}, total true reward:{}".format(self.current_number_of_lives, self.total_true_reward)
#         # super(MontezumaRevengeFogOfWarRandomAction, self)._step(action)
#
#     def _step(self, a):
#         action = a
#         if random.uniform(0, 1) < 0.05:
#             action = random.randint(0, 17)
#
#         # return super(MontezumaRevengeFogOfWarRandomAction, self)._step(action)
#
#         ob, reward, game_over, d = super(MontezumaRevengeFogOfWarRandomAction, self)._step(action)
#
#         if reward > 50:
#             self.total_true_reward += reward
#             self.state_checkpoints.append(self.createCheckPoint())
#
#         print "Observation:{}".format(ob)
#         return ob, reward, game_over, d
#
#     def _reset(self):
#          to_be_restored = random.sample(self.state_checkpoints, 1)[0]
#          self.restoreCheckPoint(to_be_restored)
#
#          return self._get_image()



