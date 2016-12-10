import tensorflow as tf
import cv2
import pong
import numpy as np
import random
from collections import deque


ACTIONS = 3

GAMMA = 0.99

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05

EXPLORE = 500000
OBSERVE = 50000

REPLAY_MEMORY = 50000

BATCH = 100
flags = tf.app.flags

eval_dir = '/Users/anatoly/Desktop/pong_neural_net_progress/'

flags.DEFINE_string("data_dir", "/Users/anatoly/Desktop/pong_neural_net_progress/", "The name of data directory [data]")
FLAGS = flags.FLAGS

save_path = 'checkpoints/bestvalidation'

def create_graph():
    w_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    w_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    w_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64]))

    w_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    w_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    s = tf.placeholder('float', [None, 84, 84, 4])

    conv1 = tf.nn.relu(tf.nn.conv2d(s, w_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, w_fc4) + b_fc4)

    fc5 = tf.matmul(fc4, w_fc5) + b_fc5
    return s, fc5


def train_graph(inp, out, sess, load=True):
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None])

    action = tf.reduce_sum(tf.mul(out, argmax), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(action - gt))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game = pong.PongGame()

    d = deque()

    frame = game.get_present_frame()
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)

    inp_t = np.stack((frame, frame, frame, frame), axis=2)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    if not load:
        ckpt = tf.train.get_checkpoint_state(FLAGS.data_dir)
        print(ckpt)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restored')

    t = 0
    epsilon = INITIAL_EPSILON

    while 1:

        out_t = out.eval(feed_dict={inp: [inp_t]})[0]

        argmax_t = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            max_index = random.randrange(ACTIONS)
        else:
            max_index = np.argmax(out_t)
        argmax_t[max_index] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        reward_t, frame = game.get_next_frame(argmax_t)
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)

        d.append((inp_t, argmax_t, reward_t, inp_t1))

        if len(d) > REPLAY_MEMORY:
            d.popleft()

        if t > OBSERVE:

            minibatch = random.sample(d, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict={inp: inp_t1_batch})

            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            train_step.run(feed_dict={
                gt: gt_batch,
                argmax: argmax_batch,
                inp: inp_batch
            })

        inp_t = inp_t1
        t += 1

        if t % 10000 == 0:
            checkpoint_path = FLAGS.data_dir + 'model.ckpt'
            saver.save(sess, checkpoint_path, global_step=t)
        if t % 100 == 0:
            print('game.tally =', game.tally)
            print("TIMESTEP", t, "| EPSILON", epsilon, "| ACTION", max_index, "| REWARD", reward_t,
                  "| Q_MAX %e" % np.max(out_t))


def main():
    sess = tf.InteractiveSession()
    inp, out = create_graph()
    train_graph(inp, out, sess, False)


main()
