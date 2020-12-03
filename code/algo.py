import numpy as np
from abc import abstractmethod
import cv2


class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


def convert_obs(obs):
    new_obs = str(obs)
    return new_obs


class MyQAgent(QAgent):
    def __init__(self, lr=0.1, discount=0.8, action_num=4, replay_size=100):
        super(MyQAgent, self).__init__()
        self.lr = lr
        self.discount = discount
        self.action_num = action_num
        self.replay_size = replay_size

        self.q_table = {}
        self.replay_pool = []

    def create_actions(self, state):
        state = convert_obs(state)
        if state not in self.q_table.keys():
            actions = np.zeros(self.action_num)
            self.q_table[state] = actions

    def select_action(self, ob):
        ob = convert_obs(ob)
        if ob not in self.q_table.keys():
            self.create_actions(ob)
            return np.random.randint(self.action_num)
        return np.argmax(self.q_table[ob])

    def update_table(self, s, a, sp, r):
        s = convert_obs(s)
        sp = convert_obs(sp)
        self.create_actions(s)
        self.create_actions(sp)
        self.q_table[s][a] += self.lr * (r + self.discount * np.max(self.q_table[sp]) - self.q_table[s][a])

        for s, a, sp, r in self.replay_pool:
            self.q_table[s][a] += self.lr * (r + self.discount * np.max(self.q_table[sp]) - self.q_table[s][a])

        if len(self.replay_pool) < self.replay_size:
            self.replay_pool.append((s, a, sp, r))
        elif self.replay_size is not 0:
            self.replay_pool[np.random.randint(self.replay_size)] = (s, a, sp, r)
