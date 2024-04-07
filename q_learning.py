# !pip install mss
# !pip install tkinter
# !pip install IPython
# !python -m pip uninstall rl --yes
# !pip install pygame
#
# !pip install keras-rl2
# !pip install pyautogui


from env2 import MinesweeperEnv




# In[3]:


import sys
from six import StringIO
import random
from random import randint

import numpy as np
import gym
from gym import spaces

# default : easy board
BOARD_SIZE = 10
NUM_MINES = 6

# cell values, non-negatives indicate number of neighboring mines
MINE = -1
CLOSED = -2


def board2str(board, end='\n'):
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            s += str(board[x][y]) + '\t'
        s += end
    return s[:-len(end)]


def is_new_move(my_board, x, y):
    """ return true if this is not an already clicked place"""
    return my_board[x, y] == CLOSED


def is_valid(x, y):
    """ returns if the coordinate is valid"""
    return (x >= 0) & (x < BOARD_SIZE) & (y >= 0) & (y < BOARD_SIZE)


def is_win(my_board):
    """ return if the game is won """
    return np.count_nonzero(my_board == CLOSED) == NUM_MINES


def is_mine(board, x, y):
    """return if the coordinate has a mine or not"""
    return board[x, y] == MINE


def place_mines(board_size, num_mines):
    """generate a board, place mines randomly"""
    mines_placed = 0
    board = np.zeros((board_size, board_size), dtype=int)
    while mines_placed < num_mines:
        rnd = randint(0, board_size * board_size)
        x = int(rnd / board_size)
        y = int(rnd % board_size)
        if is_valid(x, y):
            if not is_mine(board, x, y):
                board[x, y] = MINE
                mines_placed += 1
    return board

def to_s(row, col):
    return row*col + col


# env = MinesweeperDiscreetEnv()
env = MinesweeperEnv(10, 10, 6)

# RESET
observation = env.reset()

# TEST
# print("Observation space: ", env.get_board())
# print("Shape: ", env.get_board.shape)
# print("Action: ", env.get_action())
# print("Shape: ", env.action_space.shape)

# env.draw_state()
print()
for _ in range(1000):
    # print("state: \n", env.state_im)
    env.render('window')
    action = env.get_action()
    state, reward, done, info = env.step(action)

    print("Action", action)
    print(f"Reward: {reward} Done: {done}")
    if done:
        print("Game Finished!")
        break
# print("\nObeservation: \n", state)
# env.close()
# env.draw_state(env.state_im)
env.render('window')
env.window.close(True)

# RESET
env.reset()

# Parameter tuning

total_episodes = 10000
learning_rate = [0.7]
# Max steps per episode
max_steps = 99
# Discounting rate
gamma = 0.95
# Exploration rate
epsilon = 1.0
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
# Exponential decay rate for exploration prob
decay_rate = 0.005

qtable = {}

# List of rewards
rewards = []

for episode in range(total_episodes):
    action_size = env.nrows * env.ncols

    # Reset the environment
    state = env.reset()
    state_str = board2str(state)

    # Is this state seen? If not, add it to qtable and initialize the action array to 0
    if not state_str in qtable:
        qtable[state_str] = np.zeros(action_size)

    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        state_str = board2str(state)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:

            # action = np.argmax(qtable[flattened_state, :])
            # print(exp_exp_tradeoff, "action", action)
            action = np.argmax(qtable[state_str])


        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            # print("action random", action)

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)
        new_state_str = board2str(new_state)
        if not new_state_str in qtable:
            qtable[new_state_str] = np.zeros(action_size)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state

        # print("before:",qtable[state_str][action])
        qtable[state_str][action] = qtable[state_str][action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state_str]) - qtable[state_str][action])
        # print(np.max(qtable[new_state_str]))
        # print("after:",qtable[state_str][action])
        # qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        total_rewards += reward

        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)
    # print("episode:", episode, "reward:", total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print()