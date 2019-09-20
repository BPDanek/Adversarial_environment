#!/usr/bin/env python

# distinction between dqn (vanilla) and this:
# we want to formulate an addition to state, s, called "delta s", ds, which will cause the Q value of s+ds and a
# target action, a^t, to be higher than the q value for s+ds and any other action. The implication is that adding ds
# to our data will result in an action to be drawn. In cases where ds is not added/nothing is changed, we will
# behave normally.
# In order to retain the natural behaviour of a DQN controller, we will not be changing the weights or controller
# for that matter, we will instead minimize the loss between Q(s+ds, a^t) and Q(s+ds, a), where a^t is the target a,
# and a is any non-target a. Thee loss function will be (some) hinge loss: l(a,b) = max(b - (a + eps), 0), which
# will essentially enforce the condition: a >= b + eps
# An informal proof associated with the possibility this will work depends on the fact that our controller learns how
# to behave well within a certain set of input states from the possible set of states it's been trained on,
# called (here) the game-possible pixel space. This is the set of frames the game can generate under any scenario
# within the game. The game-possible pixel space is small, relative to the pixel space, which is a space containing
# all possible combinations of pixel intensities that a screen can generate. Given that ds blongs in the pixel space,
# and s belongs in the game-possible pixel space, we can say that s+ds belongs in the pixel space, which the controller
# may not know how to handle, which, if abused properly may result in a simple adversarial attack.

from __future__ import print_function
import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

# workaround for plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def adv_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.05)
    return tf.Variable(initial, trainable=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial, trainable=False)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial, trainable=False)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():

    # new vars for optimization
    initial = tf.truncated_normal(shape=[80, 80, 4], mean=(255.0 / 4), stddev=(255 * (0.01 ** 0.5)))
    delta_s = tf.Variable(
        name="added_perturbation",
        initial_value=initial,
        trainable=True)

    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # tf.convert_to_tensor([-1, 80, 80, 4])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])
    tf.expand_dims(delta_s, axis=0)
    s_opt = s + delta_s # todo: why cant we add this just like the bias
    # s_opt = conv2d(s, delta_s, 4)

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s_opt, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1, delta_s

def trainNetwork(s, readout, h_fc1, sess, writer, delta_s):

    eps = 1  # Q values are typically 10- 30
    q_noflap = tf.placeholder(shape=(1,), dtype=float)
    q_flap = tf.placeholder(shape=(1,), dtype=float)
    # hinge_loss = tf.nn.relu(q_noflap - q_flap + eps)
    hinge_loss = tf.reduce_sum(tf.nn.relu(tf.add(tf.subtract(readout[:,0], readout[:,1]), eps)))
    opt = tf.train.AdamOptimizer(1).minimize(loss=hinge_loss)
    # tf.summary.scalar(name="optimizer", tensor=opt)

    # # define the cost function
    # a = tf.placeholder("float", [None, ACTIONS])
    # y = tf.placeholder("float", [None])
    # readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    # cost = tf.reduce_mean(tf.square(y - readout_action))
    # train_step = tf.train.AdamOptimizer(1e-6).minimize(loss=cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks\
    vars = slim.get_variables_to_restore()
    variables_to_restore = slim.get_variables_to_restore(exclude=['added_perturbation', 'beta1_power_1:0', 'beta2_power_1:0'])
    saver = tf.train.Saver(var_list=variables_to_restore)
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # attack first few frames
        if (t % 10 is 0) and (t != 0):
            # generate a ds
            #delta_s.initializer # since we restore the network variables, we need to init this one separately to
                                  # save us some struggle with the saver

            # sample a minibatch to optimize on, the entire sequence of frames thus far in the program
            opt_batch = random.sample(list(D), 10)

            # get the batch variables
            s_opt_batch = [d[0] for d in opt_batch] # only take stats form opt batch
            # plt.imshow(np.ndarray(s_opt_batch)[:,:,1])
            # plt.show()
            # todo: why is this not working the way I expect it to????/ the way it has in the past
            i = 0
            loss = 100 # arbitrary init, needs to be above limit
            past_delta_s_img = np.ndarray((80, 80))
            while i < 1000 and loss >= 1:
                summary, loss, delta_s_eval = sess.run([opt, hinge_loss, delta_s], feed_dict={s : s_opt_batch})
                delta_s_img = np.reshape(delta_s_eval[:, :, 1], (-1, 80, 80, 1))
                delta_s_img = delta_s_img[0, :, :, 0]

                print("hinge loss at ", i, " : ", loss, "sq. norm", (np.linalg.norm(past_delta_s_img) - np.linalg.norm(delta_s_img)))

                past_delta_s_img = delta_s_img
                i = i + 1

            plt.imshow(delta_s_img)
            plt.show()
            plt.imshow(s_opt_batch[9][:, :, 1])
            plt.show()
            plt.imshow(delta_s_img + s_opt_batch[9][:,:,1])
            plt.show()

            # buffer = tf.summary.image(name="ds_visualize", tensor=delta_s_img)
            # writer.add_summary(buffer, t)

            saver.save(sess, 'delta_s/' + GAME + '-trial', global_step = t)

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    graph = tf.Graph()
    writer = tf.summary.FileWriter('logdir/', graph)
    sess = tf.InteractiveSession(graph=graph)
    s, readout, h_fc1, delta_s = createNetwork()
    trainNetwork(s, readout, h_fc1, sess, writer, delta_s)

def main():
    playGame()

if __name__ == "__main__":
    main()

# after digging in the tf src i think i need to better specify the subject of the optimization. maybe implict isnt the way to go.
# comment prev. src so they are up to date.