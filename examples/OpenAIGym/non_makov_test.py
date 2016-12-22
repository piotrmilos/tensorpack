import random

from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace
import numpy as np

class NonMarkovEnvironment(RLEnvironment):
    def __init__(self, x1 = 2, y1=2, x2=20, y2=20, move_reward = -0.1):
        super(NonMarkovEnvironment, self).__init__()
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.move_reward = move_reward
        self.restart_episode()


    def restart_episode(self):
        self.rewards = {}
        self.tunnels = {}
        self.end_points = [(random.randint(self.x1+2, self.x1+10), random.randint(self.y1+2, self.y1+10))]
        self.agent_position = (self.x1 + 1, self.y1 + 1)

    def action(self, action):
        assert action<=4, "The space of action is four moves."
        assert self.agent_position != None, "The agent is not present in this room."
        move_blocked = True
        newX, newY = self.agent_position
        if action == 0: #Move left
            newX = self.agent_position[0] - 1
            newY = self.agent_position[1]
        if action == 1: # Move right
            newX = self.agent_position[0] + 1
            newY = self.agent_position[1]
        if action == 2: # Move up
            newX = self.agent_position[0]
            newY = self.agent_position[1] - 1
        if action == 2:  # Move down
            newX = self.agent_position[0]
            newY = self.agent_position[1] + 1

        # Validate the move
        if newX>=self.x1 and newX<=self.x2 and newY>=self.y1 and newY<=self.y2:
            newPos = (newX, newY)
            reward = self.rewards[newPos] if newPos in self.rewards else self.move_reward
            isOver = newPos in self.end_points
            next_room = self
            if newPos in self.tunnels:
                next_room = self.tunnels[newPos][0]
                next_room.agent_position = self.tunnels[newPos][1]

            return reward, isOver
        else:
            return 0, False

    def current_state(self):
        screen_size = (30, 30)
        world = np.zeros(screen_size)
        for x in xrange(self.x1, self.x2):
            for y in xrange(self.y1, self.y2):
                world[x, y] = 1
        world[self.agent_position] = 2
        # for end_point in self.end_points:
        world[np.array(self.end_points)] = 3
        world[self.end_points] = 4
        return  world

    def get_action_space(self):
        return DiscreteActionSpace(4)


    # def add_reward(self, x, y, val):
    #     self.rewards[(x, y)] = val
    #
    # def add_tunnel(self, x, y, room):
    #     self.tunnels[(x,y)] = room
    #
    # def add_end_point(self, x, y):
    #     self.end_points.append((x,y))



