import random
import numpy as np


def select_balanced_maximum_lives(self, weights_str):
    def normalize(lst):
        s = sum(lst)
        return map(lambda x: float(x) / s, lst)

    weights_spec = map(lambda x: float(x), weights_str.split(" "))
    # print "Weights speoc:{}".format(weights_spec)
    c = self._snapshot_conn.cursor()
    c.execute("""SELECT min(a.ind), rewards_events, a.number_of_lives
                  FROM snapshots a
                  INNER JOIN (
                      SELECT reward_id, MAX(number_of_lives) max_lives
                      FROM snapshots
                      GROUP BY reward_id
                  ) b ON a.reward_id = b.reward_id AND a.number_of_lives = b.max_lives group by a.reward_id""")
    res = c.fetchall()


    indices = map(lambda x: x[0], res)
    weights = map(lambda x: weights_spec[x[1]], res)
    # print "Weights:{}".format(weights)
    probabilites = normalize(weights)
    index = np.random.choice(indices, size=1, p=probabilites)[0]
    # print "Reset selecting {} states with probs {} index:{}. Weights: {}".format(res, probabilites, index, weights)


    if len(res)>=1:
        print "Index:{}".format(index)
        for r in res:
            if r[0] == index:
                print "We have {} choices. Choosing {}, which is: {}".format(res, index, r)
    return index

def select_all(self):
    c = self._snapshot_conn.cursor()
    c.execute("SELECT ind FROM snapshots")
    res = c.fetchall()
    res = map(lambda x: x[0], res)
    i = random.sample(res, 1)[0]
    return i
