import time
import numpy as np
import random
import pandas as pd
from IPython.display import display


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

import pygame
import pygame.locals
import abc




class Visualizer(abc.ABC):

    @abc.abstractmethod
    def start(self, width, height):
        pass

class MinesweeeperVisualizer(Visualizer):
    TILE_SIZE = 16
    COLOUR_GREY = (189, 189, 189)
    #TILES_FILENAME = os.path.join(os.path.dirname(__file__), 'tiles.png')
    TILES_FILENAME = 'pics/tiles.png'
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

class MinesweeperEnv(object):
    def __init__(self, width, height, n_mines,
                 # based on https://github.com/jakejhansen/minesweeper_solver
                 rewards={'win': 1, 'lose': -1, 'progress': 0.3, 'guess': -0.3, 'no_progress': -0.3}):
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
        self.action_space = np.array(self.nrows * self.ncols)
        self.rewards = rewards

    def init_grid(self):
        board = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines

        while mines > 0:
            row, col = random.randint(0, self.nrows - 1), random.randint(0, self.ncols - 1)
            if board[row][col] != 'B':
                board[row][col] = 'B'
                mines -= 1

        return board

    def get_neighbors(self, coord):
        x, y = coord[0], coord[1]

        neighbors = []
        for col in range(y - 1, y + 2):
            for row in range(x - 1, x + 2):
                if ((x != row or y != col) and
                        (0 <= col < self.ncols) and
                        (0 <= row < self.nrows)):
                    neighbors.append(self.grid[row, col])

        return np.array(neighbors)

    def count_bombs(self, coord):
        neighbors = self.get_neighbors(coord)
        return np.sum(neighbors == 'B')

    def get_board(self):
        board = self.grid.copy()

        coords = []
        for x in range(self.nrows):
            for y in range(self.ncols):
                if self.grid[x, y] != 'B':
                    coords.append((x, y))

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

        state_im[state_im == 'U'] = -1
        state_im[state_im == 'B'] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im

    def init_state(self):
        unsolved_array = np.full((self.nrows, self.ncols), 'U', dtype='object')

        state = []
        for (x, y), value in np.ndenumerate(unsolved_array):
            state.append({'coord': (x, y), 'value': value})

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
            move = np.random.choice(np.nonzero(grid != 'B')[1])
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
        x, y = coord[0], coord[1]

        neighbors = []
        for col in range(y - 1, y + 2):
            for row in range(x - 1, x + 2):
                if ((x != row or y != col) and
                        (0 <= col < self.ncols) and
                        (0 <= row < self.nrows) and
                        ((row, col) not in processed)):

                    # prevent redundancy for adjacent zeros
                    processed.append((row, col))

                    index = state_df.index[state_df['coord'] == (row, col)].tolist()[0]

                    self.state[index]['value'] = self.board[row, col]

                    # recursion in case neighbors are also 0
                    if self.board[row, col] == 0.0:
                        self.reveal_neighbors((row, col), clicked_tiles=processed)

    def get_action(self):
        board = self.state_im.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x == -0.125]

        rand = np.random.random()  # random value b/w 0 & 1

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

        if self.state[action_index]['value'] == 'B':  # if lose
            reward = self.rewards['lose']
            done = True

        elif np.sum(new_state_im == -0.125) == self.n_mines:  # if win
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1

        elif np.sum(self.state_im == -0.125) == np.sum(current_state == -0.125):
            reward = self.rewards['no_progress']

        else:  # if progress
            if all(t == -0.125 for t in neighbors):  # if guess (all neighbors are unsolved)
                reward = self.rewards['guess']

            else:
                reward = self.rewards['progress']
                self.n_progress += 1  # track n of non-isoloated clicks

        return self.state_im, reward, done, {}