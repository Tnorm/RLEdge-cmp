import numpy as np
import gym
from gym import wrappers
import tensorflow as tf
import json, sys, os
from os import path
import random
from collections import deque


env_to_use = 'LunarLander-v2'





# Adapted from https://gist.github.com/heerad/d2b92c2f3a83b5e4be395546c17b274c

gamma = 0.99  # reward discount factor
h1 = 512  # hidden layer 1 size
h2 = 512  # hidden layer 2 size
h3 = 512  # hidden layer 3 size
lr = 5e-5  # learning rate
lr_decay = 1  # learning rate decay (per episode)
l2_reg = 1e-6  # L2 regularization factor
dropout = 0  # dropout rate (0 = no dropout)
num_episodes = 2  # number of episodes
max_steps_ep = 10000  # default max number of steps per episode (unless env has a lower hardcoded limit)
slow_target_burnin = 1000  # number of steps where slow target weights are tied to current network weights
update_slow_target_every = 100  # number of steps to use slow target as target before updating it to latest weights
train_every = 1  # number of steps to run the policy (and collect experience) before updating network weights
replay_memory_capacity = int(1e6)  # capacity of experience replay memory
minibatch_size = 1024  # size of minibatch from experience replay memory for updates
epsilon_start = 1.0  # probability of random action at start
epsilon_end = 0.05  # minimum probability of random action after linear decay period
epsilon_decay_length = 1e5  # number of steps over which to linearly decay epsilon
epsilon_decay_exp = 0.97  # exponential decay rate after reaching epsilon_end (per episode)
load_model = 1 # loads a checkpoint model if the value is 1


env = gym.make(env_to_use)
state_dim = np.prod(np.array(env.observation_space.shape))  # Get total number of dimensions in state
n_actions = env.action_space.n  # Assuming discrete action space


tf.reset_default_graph()

# placeholders
state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])  # input to Q network
next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])  # input to slow target network
action_ph = tf.placeholder(dtype=tf.int32, shape=[None])  # action indices (indices of Q network output)
reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])  # rewards (go into target computation)
is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None])  # indicators (go into target computation)
is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

# episode counter
episodes = tf.Variable(0.0, trainable=False, name='episodes')
episode_inc_op = episodes.assign_add(1)


def generate_network(s, trainable, reuse):
    hidden = tf.layers.dense(s, h1, activation=tf.nn.relu, trainable=trainable, name='dense', reuse=reuse)
    hidden_drop = tf.layers.dropout(hidden, rate=dropout, training=trainable & is_training_ph)
    hidden_2 = tf.layers.dense(hidden_drop, h2, activation=tf.nn.relu, trainable=trainable, name='dense_1', reuse=reuse)
    hidden_drop_2 = tf.layers.dropout(hidden_2, rate=dropout, training=trainable & is_training_ph)
    hidden_3 = tf.layers.dense(hidden_drop_2, h3, activation=tf.nn.relu, trainable=trainable, name='dense_2',
                               reuse=reuse)
    hidden_drop_3 = tf.layers.dropout(hidden_3, rate=dropout, training=trainable & is_training_ph)
    action_values = tf.squeeze(
        tf.layers.dense(hidden_drop_3, n_actions, trainable=trainable, name='dense_3', reuse=reuse))
    return action_values



with tf.variable_scope('q_network') as scope:
    # Q network applied to state_ph
    q_action_values = generate_network(state_ph, trainable=True, reuse=False)
    # Q network applied to next_state_ph (for double Q learning)
    q_action_values_next = tf.stop_gradient(generate_network(next_state_ph, trainable=False, reuse=True))

# slow target network
with tf.variable_scope('slow_target_network', reuse=False):
    # use stop_gradient to treat the output values as constant targets when doing backprop
    slow_target_action_values = tf.stop_gradient(generate_network(next_state_ph, trainable=False, reuse=False))

# isolate vars for each network
q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
slow_target_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_network')

# update values for slowly-changing target network to match current critic network
update_slow_target_ops = []
for i, slow_target_var in enumerate(slow_target_network_vars):
    update_slow_target_op = slow_target_var.assign(q_network_vars[i])
    update_slow_target_ops.append(update_slow_target_op)

update_slow_target_op = tf.group(*update_slow_target_ops, name='update_slow_target')

# Q-learning targets y_i for (s,a) from experience replay
# = r_i + gamma*Q_slow(s',argmax_{a}Q(s',a)) if s' is not terminal
# = r_i if s' terminal
# Note that we're using Q_slow(s',argmax_{a}Q(s',a)) instead of max_{a}Q_slow(s',a)
# to address the maximization bias problem via Double Q-Learning
targets = reward_ph + is_not_terminal_ph * gamma * \
                      tf.gather_nd(slow_target_action_values, tf.stack(
                          (tf.range(minibatch_size), tf.cast(tf.argmax(q_action_values_next, axis=1), tf.int32)),
                          axis=1))

# Estimated Q values for (s,a) from experience replay
estim_taken_action_vales = tf.gather_nd(q_action_values, tf.stack((tf.range(minibatch_size), action_ph), axis=1))



# saver
saver = tf.train.Saver()

sess = tf.Session()

if load_model:
    saver.restore(sess, "tmp/mdl700.ckpt")
    print("Model loaded.")
else:
    sess.run(tf.global_variables_initializer())


total_steps = 0
epsilon = 0.0


for ep in range(num_episodes):
    total_reward = 0
    steps_in_ep = 0

    observation = env.reset()

    for t in range(max_steps_ep):

        # choose action according to epsilon-greedy policy wrt Q
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            q_s = sess.run(q_action_values,
                           feed_dict={state_ph: observation[None], is_training_ph: False})
            action = np.argmax(q_s)

        # take step
        next_observation, reward, done, _info = env.step(action)
        env.render()
        total_reward += reward

        if done:
            break

        observation = next_observation
        total_steps += 1
        steps_in_ep += 1


    print(total_reward, steps_in_ep)
