import tensorflow.compat.v1 as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map

tf.set_random_seed(1)
np.random.seed(1)
tf.compat.v1.disable_eager_execution()


class DQN:
    def __init__(self, nstate, naction, LR=0.0012, DECAY=0.001, GAMMA=0.98):
        self.nstate = nstate
        self.naction = naction
        self.sess = tf.Session()
        self.memcnt = 0
        self.BATCH_SIZE = 64
        # self.LR = 0.0012  # learning rate
        self.LR = LR
        # self.EPSILON = 0.92  # greedy policy
        self.EPSILON = 1.0
        self.MAX_EPSILON = 1.0
        self.MIN_EPSILON = 0.1
        self.DECAY_RATE = DECAY
        # self.GAMMA = 0.9999  # reward discount
        self.GAMMA = GAMMA
        self.MAX_STEPS = 500
        self.MEM_CAP = 10_000
        self.mem = np.zeros((self.MEM_CAP, self.nstate * 2 + 2))  # initialize memory
        self.updataT = 150

        self.state = tf.placeholder(tf.float64, [None, self.nstate])
        self.action = tf.placeholder(tf.int32, [None, ])
        self.reward = tf.placeholder(tf.float64, [None, ])
        self.s_ = tf.placeholder(tf.float64, [None, self.nstate])

        with tf.variable_scope('q'):  # evaluation network
            l_eval = tf.layers.dense(self.state, 10, tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(0, 0.1))
            self.q = tf.layers.dense(l_eval, self.naction, kernel_initializer=tf.random_normal_initializer(0, 0.1))

        with tf.variable_scope('q_next'):  # target network, not to train
            l_target = tf.layers.dense(self.s_, 10, tf.nn.relu, trainable=False)
            q_next = tf.layers.dense(l_target, self.naction, trainable=False)

        q_target = self.reward + self.GAMMA * tf.reduce_max(q_next, axis=1)  # q_next:  shape=(None, naction),
        a_index = tf.stack([tf.range(self.BATCH_SIZE, dtype=tf.int32), self.action], axis=1)
        q_eval = tf.gather_nd(params=self.q, indices=a_index)
        loss = tf.losses.mean_squared_error(q_target, q_eval)
        self.optimizer = tf.train.AdamOptimizer(self.LR).minimize(loss)
        self.sess.run(tf.global_variables_initializer())

    # the action selection logic for the DQN agent
    def choose_action(self, status):
        fs = np.zeros((1, self.nstate))
        fs[0, status] = 1.0  # ONE HOT


        exp_exp_tradeoff = np.random.uniform(0, 1)
        if exp_exp_tradeoff > self.EPSILON:
            action = np.argmax(self.sess.run(self.q, feed_dict={self.state: fs}))
        else:
            action = np.random.randint(0, self.naction)
        return action

    def learn(self):
        if self.memcnt % self.updataT == 0:
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        rand_indexs = np.random.choice(self.MEM_CAP, self.BATCH_SIZE, replace=False)
        temp = self.mem[rand_indexs]

        bs = temp[:, 0:self.nstate]  # .reshape(self.BATCH_SIZE,NSTATUS)
        ba = temp[:, self.nstate]
        br = temp[:, self.nstate + 1]
        bs_ = temp[:, self.nstate + 2:]  # .reshape(self.BATCH_SIZE,NSTATUS)

        feed = {self.state: bs, self.action: ba, self.reward: br, self.s_: bs_}
        self.sess.run(self.optimizer, feed_dict=feed)

    def storeExp(self, s, a, r, s_):
        fs = np.zeros(self.nstate)
        fs[s] = 1.0  # ONE HOT
        fs_ = np.zeros(self.nstate)
        fs_[s_] = 1.0  # ONE HOT
        self.mem[self.memcnt % self.MEM_CAP] = np.hstack([fs, a, r, fs_])
        self.memcnt += 1

    def run_train(self, episodes):
        cnt_win = 0
        all_r = 0.0
        win_rate = []
        last_50_eps = 0
        for episode in range(episodes):
            s = env.reset()
            s = s[0]
            steps = 0
            done = False
            while not done and steps < self.MAX_STEPS:
                a = self.choose_action(s)
                s_, r, term, trunc, info = env.step(a)

                if r == 1:
                    r = 5
                    all_r += 1
                    cnt_win += 1.0
                    last_50_eps += 1

                # push action and results to memory cache
                self.storeExp(s, a, r, s_)
                # only learn if the memory is full?
                if self.memcnt > self.MEM_CAP:
                    self.learn()

                s = s_
                steps += 1
                done = term or trunc

            if episode % 50 == 0:
                print(f"Episode: {episode} - Steps: {steps} - Wins: {cnt_win}({cnt_win / 50})")
                if cnt_win / 50 >= 0.4:
                    self.EPSILON = self.EPSILON - 0.01
                elif cnt_win / 50 >= 0.2:
                    self.EPSILON = self.EPSILON - 0.005
                elif cnt_win / 50 >= 0.1:
                    self.EPSILON = self.EPSILON - 0.003
                elif cnt_win / 50 >= 0.05:
                    self.EPSILON = self.EPSILON - 0.001
                elif cnt_win / 50 >= 0.01:
                    # self.EPSILON = self.EPSILON - 0.001
                    self.EPSILON = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * np.exp(
                        -self.DECAY_RATE * episode)

                # clamp epsilon to a minimum of 0.1
                self.EPSILON = max(0.1, self.EPSILON)

                print(f"Epsilon: {self.EPSILON}")

                print("current accuracy: %.2f %%" % (cnt_win / 50.0 * 100))
                win_rate.append(cnt_win / 50)
                cnt_win = 0
                print("Global accuracy : %.2f %%" % (all_r / (episode + 1) * 100))

        print("Global accuracy : ", all_r / episodes * 100, "%")

        saver = tf.train.Saver()
        saver.save(self.sess, "test_model_8x8")

        plt.plot(win_rate)
        plt.show()

        # test the trained model by running the environment and using the model to predict actions
    def run_test(self, episodes):
        for ep in range(episodes):  # episode loop
            s = env.reset()[0]  # reset env for new episode
            done = False  # flag for detecting end of episode
            while not done:  # loop until we either find the goal or fail (fall into hole)
                a = self.choose_action(s)  # choose action from model prediction
                s, r, term, trunc, info = env.step(a)  # step agent
                done = term or trunc  # check if we need to stop this episode
                if done and r == 1:  # if we need to stop episode, and we found the goal, print debug
                    print("GOAL")


# map_8 = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
# map_8 = generate_random_map(size=8, p=0.8)
# print(map_8)

# 8x8 Standard Map: About 80% accuracy
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
env = env.unwrapped
dqn = DQN(env.observation_space.n, env.action_space.n, DECAY=0.00025, GAMMA=0.99)
dqn.run_train(10000)
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
dqn.run_test(100)
