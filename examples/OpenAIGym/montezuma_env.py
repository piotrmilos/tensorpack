import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

import random

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import sqlite3
from gym.envs.atari import AtariEnv
import numpy as np
import collections
from PIL import Image
import os
import sys

from gym.envs.atari.atari_env import to_ram

LOGS_DIR = "/mnt/storage_codi/piotr.milos/Logs"
FOG_OF_WAR_REWARD = 1
TRUE_REWARD_FACTOR = 5
DEATH_REWARD = - 200
REWARDS_FILE = "rewards.txt"

def combine_images(img1, img2):
    w1, h1, _ = img1.shape
    w2, h2, _ = img2.shape

    print "IMG1:{}".format(type(img1))
    print "IMG2:{}".format(type(img2))

    w3 = w1+w2
    h3 = max(h1, h2)

    joint_image_buf = np.full((w3, h3, 3), 255, dtype="uint8")

    joint_image_buf[0:w1, 0:h1, :] = img1
    joint_image_buf[w1:w1+w2, 0:h2, :] = img2

    return joint_image_buf


def get_debug_graph(values):
    sns.set(style="white", context="talk")

    plt.figure(figsize=(5, 2))
    x = np.array(['no', 'f', 'u', 'r', 'l', 'd', 'ur', 'ul', 'dr',
                  'dl', 'uf', 'rf', 'lf', 'df', 'urf', 'drf', 'dlf', '', 'val'])
    y1 = values
    sns.barplot(x, y1, palette="BuGn_d")
    sns.despine(bottom=True)

    plt.savefig("/tmp/aaa.png")

    import matplotlib.image as mpimg
    arr = np.array(Image.open("/tmp/aaa.png"))
    buf = arr[:,:,:3]

    return buf

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
        self.worker_id = random.randint(1000, 9999)
        self._rewards_events_number_of_lives = [0]*10
        self._is_important_event = 0
        self._visited_levels_mask = [0] * 100
        self._visited_levels_mask[0] = 1
        self._internal_index = 0
        self._old_fog_of_war = None
        self._maximal_agent_rewards = 0
        try:
            self.conn = sqlite3.connect(os.path.join(LOGS_DIR, 'montezuma_logs.db'))
        except:
            self.conn = None
        self.experiment_id = None
        self._reward_hash = 0

        # # self._filter_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 33, 35, 39, 41, 46, 57, 59, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 85, 98, 104, 105, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]
        # self._filter_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 33, 39, 41, 57, 59, 69, 71, 72, 73, 74, 75, 76, 77, 78, 85, 98, 104, 105, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]
        # self._filter_list = [8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 33, 39, 41, 57, 59, 69, 71, 72, 73, 74, 75, 76, 77, 78, 85, 98, 104, 105, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]
        # # self._filter_list = [x for x in xrange(128)]
        # self._board_destection = [set([]) for x in xrange(len(self._filter_list))]
        self.counter = 0
        # print "Creating a Montezuma agent with id:{}".format(self._internal_id)


    def set_experiment_id(self, exp_id):
        self.experiment_id = exp_id

    def insert_logs(self, *args):
        if self.conn == None:
            #We do not log
            return
        assert len(args) == 12, "wrong number of parameters"
        sql_instert_str = "INSERT INTO Logs ('experiment_id', 'worker_id', 'image_id', " \
                          "'event_type', 'rewards_events', 'number_of_lives', 'number_of_levels', " \
                          "'level0', 'level1', 'level2', 'level3', 'level4') " \
                          "VALUES ({},{},'{}',{},{},{},{},{},{},{},{},{})".format(*args)
        # print "SQL command:{}".format(sql_instert_str)
        cursor = self.conn.cursor()
        cursor.execute(sql_instert_str)
        self.conn.commit()



    def log_important_event(self, image):
        if self.conn == None:
            return
        data = np.lib.pad(image, ((0, 0), (0, 300), (0, 0)), 'constant', constant_values=(0, 255))
        img = Image.fromarray(data)
        draw = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/lyx/cmmi10.ttf"
        font = ImageFont.truetype(font_path, size=15)

        debug_string = "Debug info for worker:{}\n".format(self.worker_id)

        vec = [self._rewards_events, sum(self._visited_levels_mask)] + self._rewards_events_number_of_lives[:3] + self._visited_levels_mask[:5]
        descs = ["rewards", "levels", "lives0", "lives1", "lives2",
                 "level0", "level1", "level2", "level3", "level4"]
        assert len(vec)==len(descs), "Oh shit:{}, {}!".format(len(vec), len(descs))
        for val, desc in zip(vec, descs):
            debug_string += "   {}:{}\n".format(desc, val)
        draw.text((180, 0), debug_string, (0, 0, 0), font=font)

        image_str = "screenshot{}-{}-{}.png".format(self.experiment_id, self.worker_id, self._internal_index)
        self._internal_index += 1

        img.save(os.path.join(LOGS_DIR, image_str))

        self.insert_logs(self.experiment_id, self.worker_id, image_str,
                         self._is_important_event, self._rewards_events, self.current_number_of_lives, sum(self._visited_levels_mask),
                         self._visited_levels_mask[0], self._visited_levels_mask[1], self._visited_levels_mask[2],
                         self._visited_levels_mask[3], self._visited_levels_mask[4])

        self._is_important_event = 0

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        print "AAA"
        screen_img = self._get_image()
        debug_img = get_debug_graph([1 for i in xrange(1,20)])

        img = combine_images(screen_img, debug_img)

        # img = debug_img


        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)


    def _get_image(self):

        self.ale.getScreenRGB(self._buffer)  # says rgb but actually bgr
        res = self._buffer[:, :, [2, 1, 0]]


        fog_of_war = np.kron(self._fogs_of_war[self.board_id], self.upsampling_matrix)
        res[:,:,2] = fog_of_war


        log_information = self._information_to_encode()
        info_size = len(log_information)
        info = np.lib.pad(np.array(log_information),
                          (0, 160 - info_size), 'constant', constant_values=(0,0))

        res[0, :, 0] = info


        return res

    def _information_to_encode(self):
        l = []
        # print "Montezuma rewards:{}".format(self._rewards_events)
        l.append(self._rewards_events)
        l.append(sum(self._visited_levels_mask))

        l += self._rewards_events_number_of_lives[:3]
        l += self._visited_levels_mask[:7]


        return l

    def _locate_agent(self):
        HACK_PARAMETER = 30
        positions = np.argwhere(self._buffer[HACK_PARAMETER:,:,2]==200)

        if len(positions) != 0:
            return (positions[0,0]+ HACK_PARAMETER, positions[0,1])


    # def set_reward_vector(self, vec):
    #     self.reward_



    def _step(self, a):
        global TRUE_REWARD_FACTOR, DEATH_REWARD, FOG_OF_WAR_REWARD

        loc = self._locate_agent()

        ob, reward, game_over, d = super(MontezumaRevengeFogOfWar, self)._step(a)
        reward = TRUE_REWARD_FACTOR * reward

        self._set_board(ob)

        if reward != 0:
            self._old_fog_of_war = self._fogs_of_war[self.board_id]
            self._fogs_of_war[self.board_id] = np.ones((21,16), dtype="uint8")
            self._rewards_events_number_of_lives[self._rewards_events] = self.current_number_of_lives
            self._rewards_events += 1
            self._reward_hash = hash((self.board_id, self._rewards_events, loc[0]/20, loc[1]/20))
            print "Registered {} on {} {} with hash {}".format(self._rewards_events, loc[0]/20, loc[1]/20, self._reward_hash)

            if self._rewards_events>self._maximal_agent_rewards:
                self._is_important_event = 1
                self._maximal_agent_rewards = self._rewards_events




        if self._number_of_lives(ob) != self.current_number_of_lives:
            self.current_number_of_lives = self._number_of_lives(ob)
            reward += DEATH_REWARD
            # self._is_important_event = 2




        if loc != None:
            # print "Agent location:{}".format(loc)
            l_x = loc[0]/10
            l_y = loc[1]/10
            if self._fogs_of_war[self.board_id][l_x, l_y] == 1:
                reward += FOG_OF_WAR_REWARD
                self._fogs_of_war[self.board_id][l_x, l_y] = 0



        if self._is_important_event > 0:
            self.log_important_event(ob)

        return ob, reward, game_over, d

    def _set_board(self, ob):
        _detected_board = -1

        tmp_sum = np.sum(ob[:, 1, 0])

        # print "Sum1:{} and sum2:{}".format(np.sum(ob[:, 159, 0]), sum)

        # print "Sum:{}".format(np.sum(ob[:, 1, 0]))

        if np.all(ob[94:133, 159, 1] == 158):
            _detected_board = 0
        if np.sum(ob[:, 159, 0]) == 7746:
            _detected_board = 1
        if abs(np.sum(ob[:, 1, 0]) - 9216) <= 10:
            _detected_board = 2
        if np.sum(ob[100:, 80, 0]) == 14080:
            _detected_board = 3
        if abs(tmp_sum - 12227) <= 3:
            _detected_board = 4
        if abs(np.sum(ob[:, 159, 0]) - 19561) <= 3:
            _detected_board = 5
        if abs(np.sum(ob[:, 1, 0]) - 6845) <= 2:
            _detected_board = 6
        if abs(tmp_sum - 200) <= 3 or abs(tmp_sum - 314) <= 3:
            _detected_board = self.board_id

        if _detected_board != -1:
            self.board_id = _detected_board

        old_number_of_visited = sum(self._visited_levels_mask)
        self._visited_levels_mask[self.board_id] = 1
        if sum(self._visited_levels_mask) != old_number_of_visited:
            self._is_important_event = 4

        if _detected_board == -1:
            self._is_important_event = 8

        # ram = self._get_ram()
        # for x in xrange(len(self._filter_list)):
        #     self._board_destection[x].add((ram[self._filter_list[x]], self.board_id))
        #
        #
        # suspects = []
        # for x in xrange(len(self._filter_list)):
        #     if len(self._board_destection[x]) == sum(self._visited_levels_mask):
        #         suspects.append(self._filter_list[x])
        # if len(suspects)<len(self._filter_list):
        #     print "Suspects {}:{}".format(len(suspects), suspects)


    def _set_board_old(self, ob):
        ram = self._get_ram()
        print "Ram:{} and board:{}".format(ram, ram[83])

        self.board_id = ram[83]

        old_number_of_visited = sum(self._visited_levels_mask)
        self._visited_levels_mask[self.board_id] = 1
        if sum(self._visited_levels_mask) != old_number_of_visited:
            self._is_important_event = 4
            print "Imporatant event"
            raise AttributeError


    def _reset(self):
        self._read_rewards()

        for x in xrange(10):
            self._fogs_of_war[x] = np.ones((21,16), dtype="uint8")
        self.current_number_of_lives = 5
        self._rewards_events_number_of_lives = [0] * 10
        self._rewards_events = 0
        self._boards_explored = 1
        return super(MontezumaRevengeFogOfWar, self)._reset()

    def _number_of_lives(self, observation):
        return int(np.sum(observation[15, :, 0])) / (3 * 210)

    def _read_rewards(self):
        global FOG_OF_WAR_REWARD, TRUE_REWARD_FACTOR, DEATH_REWARD
        try:
            with open(REWARDS_FILE, "r") as f:
                rewards = f.readline()
                strs = rewards.split(" ")
                TRUE_REWARD_FACTOR = float(strs[0])
                FOG_OF_WAR_REWARD = float(strs[1])
                DEATH_REWARD = float(strs[2])
        except:
            TRUE_REWARD_FACTOR, FOG_OF_WAR_REWARD, DEATH_REWARD = 5, 1, -200

        # print "New rewards have been set: {} {} {}".format(TRUE_REWARD_FACTOR, FOG_OF_WAR_REWARD, DEATH_REWARD)




class MontezumaRevengeFogOfWarRandomAction(MontezumaRevengeFogOfWar):

    def _step(self, a):
        # print "{}".format(self._internal_id)
        action = a
        if random.uniform(0,1) < 0.00:
            action = random.randint(0,17)

        return super(MontezumaRevengeFogOfWarRandomAction, self)._step(action)


STARTING_POINT_SELECTOR = "/mnt/storage_codi/piotr.milos/Projects/rl2/" \
                          "tensorpack/examples/OpenAIGym/starting_point_selector"

class MontezumaRevengeFogOfWarRandomActionRandomStartingState(MontezumaRevengeFogOfWar):

    def __init__(self, game='pong', obs_type='ram', frameskip=(2, 5), repeat_action_probability=0.):
        super(MontezumaRevengeFogOfWarRandomActionRandomStartingState, self).__init__(game, obs_type, frameskip, repeat_action_probability)
        self._saved_states = []
        self.setup_database()
        self._save_current_state(0)
        sys.path.insert(0, "/mnt/storage_codi/piotr.milos/Projects/rl2/tensorpack/examples/OpenAIGym")


        #########self._saved_states.append(self._save_current_state())

    def _save_current_state(self, reward_id):
        t= (self.ale.cloneState(), self._fogs_of_war, self.current_number_of_lives, self._rewards_events,
            self._rewards_events_number_of_lives)
        self.insert_row(reward_id, self._rewards_events, self.current_number_of_lives, len(self._saved_states))
        self._saved_states.append(t)

    def _restore_state(self, state):
        t0, t1, t2, t3, t4 = state
        self.ale.restoreState(t0)
        self._fogs_of_war = {}
        for id in xrange(10):
            self._fogs_of_war[id] = np.copy(t1[id])
        self.current_number_of_lives = t2
        self._rewards_events = t3
        self._rewards_events_number_of_lives = list(t4)
        # self._visited_levels_mask = list(t5)
        self._is_important_event = 0


    def _step(self, a):
        ob, reward, game_over, d = super(MontezumaRevengeFogOfWarRandomActionRandomStartingState, self)._step(a)

        if self._reward_hash != 0:
            self._save_current_state(self._reward_hash)
            self._reward_hash = 0

        return ob, reward, game_over, d

    def _reset(self):
        self._read_rewards()
        try:
            import montezuma_starting_point_processors
            reload(montezuma_starting_point_processors)
            import montezuma_starting_point_processors
            # print montezuma_starting_point_processors
            # command_name = ""
            with open(STARTING_POINT_SELECTOR, "r") as f:
                selector = f.readline()
                # print "Selector:{}".format(selector)
                (command_name, command_param) = tuple(selector.split(":"))


            selector = getattr(montezuma_starting_point_processors, command_name)
            state_to_restore_index = selector(self, command_param)
        except Exception as e:
             print "Section failed. Rolling back to select_all. Error: {}".format(e)
             state_to_restore_index = montezuma_starting_point_processors.select_all(self)

        self._restore_state(self._saved_states[state_to_restore_index])

        return self. _get_image()



    def setup_database(self):
        self._saved_states = []
        self._saved_states_index = 0
        self._snapshot_conn = sqlite3.connect(":memory:")
        c = self._snapshot_conn.cursor()
        c.execute("""CREATE TABLE snapshots (
                   reward_id       INTEGER NOT NULL,
                   rewards_events  INTEGER NOT NULL,
                   number_of_lives INTEGER NOT NULL,
                   ind             INTEGER NOT NULL)""")
        self._snapshot_conn.commit()


    def insert_row(self, reward_id, rewards_events, number_of_lives, ind):
        c = self._snapshot_conn.cursor()
        c.execute("INSERT INTO snapshots VALUES ({}, {}, {}, {})".format(reward_id, rewards_events, number_of_lives, ind))
        self._snapshot_conn.commit()

