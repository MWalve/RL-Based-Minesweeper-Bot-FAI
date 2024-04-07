# In[25]:


# AUTHOR: Yichun Zhao

# !pip install mss
# !pip install tkinter
# !pip install IPython
# !python -m pip uninstall rl --yes
#!pip install pygame

#!pip install keras-rl2
#!pip install pyautogui

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[26]:


import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

import pygame
import pygame.locals
import abc
import time
import numpy as np


class Visualizer(abc.ABC):

    @abc.abstractmethod
    def start(self, width, height):
        pass

class MinesweeeperVisualizer(Visualizer):
    TILE_SIZE = 16
    COLOUR_GREY = (189, 189, 189)
    #TILES_FILENAME = os.path.join(os.path.dirname(__file__), 'tiles.png')
    TILES_FILENAME = './tiles.png'
    TILE_HIDDEN = 9
    TILE_EXPLODED = 10
    TILE_BOMB = 11
    TILE_FLAG = 12
    WINDOW_NAME = 'Minesweeper'

    def __init__(self):
        self.game_width = 0
        self.game_height = 0
        self.num_mines = 0
        self.screen = None
        self.tiles = None

    def start(self, width, height, num_mines):
        self.game_width = width
        self.game_height = height
        self.num_mines = num_mines
        pygame.init()
        pygame.mixer.quit()
        pygame.display.set_caption(self.WINDOW_NAME)
        screen_width = self.TILE_SIZE * self.game_width
        screen_height = self.TILE_SIZE * self.game_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill(self.COLOUR_GREY)
        self.tiles = self._load_tiles()

    def wait(self):
        while 1:
            event = pygame.event.wait()
            if event.type == pygame.locals.KEYDOWN:
                break
            elif event.type == pygame.locals.QUIT:
                pygame.quit()
                break

    def close(self, pause):
        if pause:
            self.wait()
        pygame.quit()

    def _load_tiles(self):
        image = pygame.image.load(self.TILES_FILENAME).convert()
        image_width, image_height = image.get_size()
        tiles = []
        for tile_x in range(0, image_width // self.TILE_SIZE):
            rect = (tile_x * self.TILE_SIZE, 0, self.TILE_SIZE, self.TILE_SIZE)
            tiles.append(image.subsurface(rect))
        return tiles

    def _draw(self, observation):
        openable = self.game_width * self.game_height - self.num_mines
        unique, counts = np.unique(observation, return_counts=True)
        unopened = dict(zip(unique, counts))[-1]
        all_opened = unopened == self.num_mines

        for x in range(self.game_width):
            for y in range(self.game_height):
                if observation[x, y] == -1:
                    if all_opened:
                        tile = self.tiles[self.TILE_BOMB]
                    else:
                        tile = self.tiles[self.TILE_HIDDEN]
                elif observation[x, y] == -2:
                    tile = self.tiles[self.TILE_EXPLODED]
                else:
                    tile = self.tiles[int(observation[x, y])]
                self.screen.blit(tile, (16 * x, 16 * y))
        pygame.display.flip()


# In[27]:


BOARD_SIZE = 4
state_size, action_size = BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE
#state_size, action_size = 64, 64

def board_completion(board):
    count_pos = 0
    num_pos = state_size
    for pos in np.nditer(board):
        #print(pos)
        if not pos == -0.125:
            count_pos = count_pos+1
    return count_pos/num_pos 


def state_to_str(state):
    s = ''
    for x in range(4):
        for y in range(4):
            s += str(state[x][y])
    return s


# In[28]:


# env 2

import random
import numpy as np
import pandas as pd
from IPython.display import display

class MinesweeperEnv(object):
    def __init__(self, width, height, n_mines,
        # based on https://github.com/jakejhansen/minesweeper_solver
        rewards={'win':1, 'lose':-1, 'progress':0.3, 'guess':0, 'no_progress' : -0.3}):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()
        self.n_clicks = 0
        self.n_progress = 0
        self.n_wins = 0
        self.window = None

        self.rewards = rewards

    def init_grid(self):
        board = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines

        while mines > 0:
            row, col = random.randint(0, self.nrows-1), random.randint(0, self.ncols-1)
            if board[row][col] != 'B':
                board[row][col] = 'B'
                mines -= 1

        return board

    def get_neighbors(self, coord):
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(self.grid[row,col])

        return np.array(neighbors)

    def count_bombs(self, coord):
        neighbors = self.get_neighbors(coord)
        return np.sum(neighbors=='B')

    def get_board(self):
        board = self.grid.copy()

        coords = []
        for x in range(self.nrows):
            for y in range(self.ncols):
                if self.grid[x,y] != 'B':
                    coords.append((x,y))

        for coord in coords:
            board[coord] = self.count_bombs(coord)

        return board

    def get_state_im(self, state):
        '''
        Gets the numeric image representation state of the board.
        This is what will be the input for the DQN.
        '''

        state_im = [t['value'] for t in state]
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=='U'] = -1
        state_im[state_im=='B'] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im

    def init_state(self):
        unsolved_array = np.full((self.nrows, self.ncols), 'U', dtype='object')

        state = []
        for (x, y), value in np.ndenumerate(unsolved_array):
            state.append({'coord': (x, y), 'value':value})

        state_im = self.get_state_im(state)

        return state, state_im

    def color_state(self, value):
        if value == -1:
            color = 'white'
        elif value == 0:
            color = 'slategrey'
        elif value == 1:
            color = 'blue'
        elif value == 2:
            color = 'green'
        elif value == 3:
            color = 'red'
        elif value == 4:
            color = 'midnightblue'
        elif value == 5:
            color = 'brown'
        elif value == 6:
            color = 'aquamarine'
        elif value == 7:
            color = 'black'
        elif value == 8:
            color = 'silver'
        else:
            color = 'magenta'

        return f'color: {color}'

    def draw_state(self, state_im):
        state = state_im * 8
        state_df = pd.DataFrame(state.reshape((self.nrows, self.ncols)), dtype=np.int8)

        display(state_df.style.applymap(self.color_state))

    def render(self, mode):
        if mode == 'human':
            self.draw_state(self.state_im)
        elif mode == 'window':
            state = self.state_im * 8
            self.window = MinesweeeperVisualizer()
            self.window.start(self.nrows, self.ncols, self.n_mines)
            self.window._draw(state)
    
    
    def click(self, action_index):
        coord = self.state[action_index]['coord']
        value = self.board[coord]

        # ensure first move is not a bomb
        if (value == 'B') and (self.n_clicks == 0):
            grid = self.grid.reshape(1, self.ntiles)
            move = np.random.choice(np.nonzero(grid!='B')[1])
            coord = self.state[move]['coord']
            value = self.board[coord]
            self.state[move]['value'] = value
        else:
            # make state equal to board at given coordinates
            self.state[action_index]['value'] = value

        # reveal all neighbors if value is 0
        if value == 0.0:
            self.reveal_neighbors(coord, clicked_tiles=[])

        self.n_clicks += 1

    def reveal_neighbors(self, coord, clicked_tiles):
        processed = clicked_tiles
        state_df = pd.DataFrame(self.state)
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows) and
                    ((row, col) not in processed)):

                    # prevent redundancy for adjacent zeros
                    processed.append((row,col))

                    index = state_df.index[state_df['coord'] == (row,col)].tolist()[0]

                    self.state[index]['value'] = self.board[row, col]

                    # recursion in case neighbors are also 0
                    if self.board[row, col] == 0.0:
                        self.reveal_neighbors((row, col), clicked_tiles=processed)

    
    def get_action(self):
        board = self.state_im.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        
        move = np.random.choice(unsolved)
        

        return move
    
    def reset(self):
        self.n_clicks = 0
        self.n_progress = 0
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()
        return self.state_im

    def step(self, action_index):
        done = False
        coords = self.state[action_index]['coord']

        current_state = self.state_im

        # get neighbors before action
        neighbors = self.get_neighbors(coords)

        self.click(action_index)

        # update state image
        new_state_im = self.get_state_im(self.state)
        self.state_im = new_state_im

        if self.state[action_index]['value']=='B': # if lose
            reward = self.rewards['lose']
            done = True

        elif np.sum(new_state_im==-0.125) == self.n_mines: # if win
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1

        elif np.sum(self.state_im == -0.125) == np.sum(current_state == -0.125):
            reward = self.rewards['no_progress']

        else: # if progress
#             if all(t==-0.125 for t in neighbors): # if guess (all neighbors are unsolved)
#                 reward = self.rewards['guess']

#             else:
                reward = self.rewards['progress']
                self.n_progress += 1 # track n of non-isoloated clicks

        return self.state_im, reward, done, {} 


# In[ ]:





# In[29]:


env = MinesweeperEnv(4,4,4)
observation = env.reset()


# In[30]:


#env = MinesweeperDiscreetEnv()
env = MinesweeperEnv(4,4,4)

# RESET
observation = env.reset()

# TEST
#print("Observation space: ", env.get_board())
#print("Shape: ", env.get_board.shape)
#print("Action: ", env.get_action())
#print("Shape: ", env.action_space.shape)

#env.draw_state()
print()
for _ in range(10):
    #print("state: \n", env.state_im)
    env.draw_state(env.state_im)
    env.state_im
    board_completion(env.state_im)
    
    action =  env.get_action()
    state, reward, done, info = env.step(action)

        
    print("Action", action)
    print(f"Reward: {reward} Done: {done}")
    if done:
        env.draw_state(env.state_im)
        env.state_im
        board_completion(env.state_im)
        print("Game Finished!")
        break
#print("\nObeservation: \n", state)
#env.close()
#env.draw_state(env.state_im)
env.render('window')
env.window.close(False)

# RESET
env.reset()


# Q Learning (adapted from https://colab.research.google.com/github/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q_Learning_with_FrozenLakev2.ipynb)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Deep Q Learning

# In[31]:


import keras
# import keras-rl2

from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent 
from rl.policy import EpsGreedyQPolicy
from rl.policy import LinearAnnealedPolicy
from rl.memory import SequentialMemory
#from tensorflow.keras.models import Sequential


# In[32]:


def naive_nn_model():
    model = Sequential()
    model.add(Flatten(input_shape=(np.insert(env.state_im.shape, 0, 1))))
    #model.add(Flatten(input_shape=(1,state_size)))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(action_size))
    
    return model


# In[33]:


def cnn_model(conv_size, dense_size):
    model = Sequential()
    model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same', input_shape=(np.insert(env.state_im.shape, 0, 1))))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2, 2),strides=(2,2)))
    model.add(Conv2D(2*conv_size, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2, 2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(action_size, activation='linear'))
    
    return model


# In[34]:


def cnn_more_layers_model(conv_size, dense_size):
    model = Sequential()
    model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same', input_shape=(np.insert(env.state_im.shape, 0, 1))))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2, 2),strides=(2,2)))
    model.add(Conv2D(2*conv_size, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2*conv_size, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2, 2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(dense_size/2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(action_size, activation='linear'))
    
    return model


# In[35]:


n=6
m=6
env = MinesweeperEnv(n,n,m)
obs = env.reset()
#print(obs)
#print(env.state_im)
state_size = n*n
action_size = n*n


#model = naive_nn_model()
#model = cnn_model(16, 128)
#model = cnn_model(64, 512)
model = cnn_more_layers_model(64, 512)
model.summary()

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='cnn_more_layers_model.png')


# In[36]:


policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.005, value_test=0.05, nb_steps=1500000)
# The limit parameter specifies how many entries (states) the memory can hold.
memory = SequentialMemory(limit=1800000, window_length=1)
dqn = DQNAgent(
               model=model,
               nb_actions = action_size, 
               memory=memory, 
               #nb_steps_warmup=1000, 
               #target_model_update=1e-3, 
               policy=policy,
               enable_double_dqn=True,
               enable_dueling_network=True,
               gamma = 0.9
              )
dqn.compile(Adam(lr=0.01), metrics=['mae'])
dqn.load_weights('saved_weights_2500000')


# In[37]:


# Results


# 1000 steps and 4x4 env


# --- with double dqn

# naive_nn: -0.2562 

# cnn 1: -0.2011

# cnn 2: -0.1959

# cnn more layer: -0.1820


# --- without double dqn

# cnn more layer: -0.1970 


# In[38]:


#dqn.save_weights('saved_weights_testrun', overwrite=True)
#dqn.load_weights('saved_weights')


# In[39]:


#dqn.test(env, nb_episodes=5, visualize=False)


# In[ ]:




total_episodes = 100       # Total episodes
learning_rate = 0.05          # Learning rate
max_steps = 100 #env.observation_space.shape[0] * env.observation_space.shape[1]              # Max steps per episode
gamma = 0.9                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.005            # Minimum exploration probability 
decay_rate = 0.01            # Exponential decay rate for exploration prob

#qtable = np.zeros((state_size, action_size))
# Q tabel is now a hash dictionary!
qtable = {}

# List of rewards
rewards = []

#%%time

# Testing the learned agent
env.reset()

# List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(100):
    print("****************************************************")
    print("EPISODE ", episode)
    # Reset the environment
    
    
    env.reset()
    done = False 

    state = env.state_im
        

    step = 0
    total_rewards = 0
    
    for step in range(max_steps):
        env.render('window')
        
        state_str = state_to_str(state)
        
        action = dqn.forward(state)
        new_state, reward, done, info = env.step(action)
        


        total_rewards += reward
           
        # If done (if we're dead) : finish episode
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            print("Done!")
            #print(env.get_board())
            # if is_win(new_state):
            #     print("We reached our Goal 馃弳")
            # else:
            #     print("We failed 鈽狅笍")
            
            print("Number of steps", step)
            print("Reward", reward)
            #percent = board_completion(new_state)
            #print("Board completion", percent)

            break
        state = new_state
        

    env.render('window')
    rewards.append(total_rewards)
    if not done:
        print("Number of steps", step)
        # percent = board_completion(state)
        # print("Board completion", percent)
    
print ("Score over time: " +  str(sum(rewards)/total_episodes))




