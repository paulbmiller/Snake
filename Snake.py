# -*- coding: utf-8 -*-
import tkinter as tk
import numpy as np
import time
import uuid
import csv
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
# from DeepQLearning import DeepQNAgent
from RL import Policy

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
GRID_SIZE = 40
H = WINDOW_HEIGHT // GRID_SIZE
W = WINDOW_WIDTH // GRID_SIZE


def to_grid(x):
    return x * GRID_SIZE


def from_grid(x):
    return x // GRID_SIZE


def random_direction():
    return randint(0, 2)


def spawn_fruit(snake, canvas):
    global FRUIT_X, FRUIT_Y, FRUIT

    while True:
        FRUIT_X = randint(0, W-1)
        FRUIT_Y = randint(0, H-1)
        if (FRUIT_X, FRUIT_Y) != snake.head \
            and (FRUIT_X, FRUIT_Y) not in map(lambda l: l.get_coords(),
                                              snake.body):
            break

    if canvas is not None:
        FRUIT = canvas.create_rectangle(to_grid(FRUIT_X), to_grid(FRUIT_Y),
                                        to_grid(FRUIT_X) + GRID_SIZE,
                                        to_grid(FRUIT_Y) + GRID_SIZE,
                                        fill="red")


def start_snake(eat, death, step, display=False, display_title='Snake'):
    if display:
        window = tk.Tk()

        window.winfo_toplevel().title(display_title)

        can = tk.Canvas(window, width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                        highlightthickness=0)

        can.config(bg="black")

        can.pack()

    else:
        window, can = None, None

    s = Snake(can, eat, death, step)

    spawn_fruit(s, can)

    return window, s, can


def init_data_files(nb_games, disc, nn_lr, pol_lr, eps0, eps1, bs, eat, death,
                    step, optim, loss_fn):
    # Create a file where we will store the game results
    unique_id = uuid.uuid4().hex
    filename = 'results//' + unique_id
    filename = filename + '.csv'
    with open(filename, 'w', newline='') as f:
        dw = csv.DictWriter(f, fieldnames=['Game_id', 'Score', 'Steps',
                                           'Epsilon'])
        dw.writeheader()

    # Add this run to an index, so that we know what runs have what params
    results_filename = 'results//results_index.csv'
    if not os.path.exists(results_filename):
        with open(results_filename, 'w', newline='') as f:
            dw = csv.DictWriter(f, fieldnames=['nb_games', 'discount',
                                               'nn_lr', 'policy_lr',
                                               'eps_start', 'eps_end',
                                               'batch_size', 'Eat_reward',
                                               'Death_Pun', 'Step_pun',
                                               'Optimizer', 'Loss_fn',
                                               'string_identification'])
            dw.writeheader()

    with open(results_filename, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([nb_games, disc, nn_lr, pol_lr, eps0, eps1, bs, eat,
                         death, step, str(optim), str(loss_fn()), unique_id])

    return filename


def store_data(filename, game_id, score, steps, epsilon):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([game_id, score, steps, epsilon])


def plot(filename, mean_every=20, column='Score', save_fig=False):
    # Plot the given column of the csv for the games
    i = 0
    new_ser = pd.Series()
    df = pd.read_csv(filename)
    df = df[column]
    title = 'Mean value of {} every {} games'.format(column.lower(),
                                                     mean_every)

    while i < len(df):
        new_ser = new_ser.append(pd.Series([df[i:i+mean_every].mean()]),
                                 ignore_index=True)
        i += mean_every
    new_ser.index = new_ser.index * mean_every
    fig = new_ser.plot(title=title, legend=False).get_figure()

    if save_fig:
        fig.savefig(filename[:-4] + column.lower() + '.pdf')


def run_user():
    window, s, can = start_snake(eat=0, death=0, step=0, display=True)
    can.bind("<Key>", s.key)
    can.focus_set()

    try:
        window.mainloop()
    except KeyboardInterrupt:
        if window:
            window.quit()
            window.destroy()


def run(display, epsilon, alpha, discount, eps_end, batch_size, nb_games, eat,
        death, step, optim, loss_fn):
    agent = Policy(epsilon=epsilon, alpha=alpha, discount=discount,
                   eps_end=eps_end, optim=optim, loss_fn=loss_fn)
    window = None
    scores = np.array([])
    i = 1
    print("Starting run")
    results_filename = init_data_files(nb_games, discount, alpha, agent.lr,
                                       epsilon, eps_end, batch_size, eat,
                                       death, step, optim, loss_fn)

    try:
        while i <= nb_games:
            str_out = "Starting game {}".format(i)
            str_out += ", epsilon : {:.3f}".format(agent.epsilon)
            print(str_out)
            if display:
                win_title = 'Snake {}'.format(i)
                window, s, can = start_snake(eat, death, step, display=True,
                                             display_title=win_title)
            else:
                window, s, can = start_snake(eat, death, step, display=False)

            while not s.dead:
                if window is not None:
                    time.sleep(0.05)

                state = s.get_state()
                action = agent.choose_action(state)
                s.step(action)
                agent.store_transition(state, action, s.reward)

                if window is not None:
                    window.update()

            if window is not None:
                time.sleep(0.1)
                window.destroy()

            agent.learn(batch_size)

            print("Score : {}, steps: {}".format(s.score, agent.steps))

            scores = np.append(scores, s.score)
            store_data(results_filename, game_id=i, score=s.score,
                       steps=agent.steps, epsilon=agent.epsilon)
            i += 1
            agent.steps = 0

        plt.figure(0)
        plot(results_filename, mean_every=20, column='Score', save_fig=True)
        plt.figure(1)
        plot(results_filename, mean_every=20, column='Steps', save_fig=True)

    except KeyboardInterrupt:
        if window is not None:
            window.destroy()
        plt.figure(0)
        plot(results_filename, mean_every=20, column='Score', save_fig=True)
        plt.figure(1)
        plot(results_filename, mean_every=20, column='Steps', save_fig=True)

    print("Sum of scores after {} games : {}".format(i-1, scores.sum()))


class Snake(object):
    def __init__(self, canvas, eat, death, step):
        self.dead = False
        self.just_ate = False
        self.size = 3   # Size of the body (does not include the head)
        self.canvas = canvas
        self.score = 0
        self.reward = 0
        self.EAT_REWARD = eat
        self.DEATH_PUNISH = death
        self.STEP_PUNISH = step

        # x and y position of the head
        self.head = (W//2, H//2)

        """Choose a random direction in which to start (0 for right, 1 for
        down, 2 for left and 3 for up) and initialize the snake going straight.
        """
        self.direction = randint(0, 3)

        # x and y positions of the body (does not include the head)
        self.body = []

        self.body.append(Body(self.head[0], self.head[1], 0, self.canvas))

        if self.direction == 0:
            for i in range(self.size):
                self.body.append(Body(self.head[0]-i-1, self.head[1], i+1,
                                      self.canvas))
        elif self.direction == 1:
            for i in range(self.size):
                self.body.append(Body(self.head[0], self.head[1]-i-1, i+1,
                                      self.canvas))
        elif self.direction == 2:
            for i in range(self.size):
                self.body.append(Body(self.head[0]+i+1, self.head[1], i+1,
                                      self.canvas))
        else:
            for i in range(self.size):
                self.body.append(Body(self.head[0], self.head[1]+i+1, i+1,
                                      self.canvas))

    def get_state(self):
        state = []

        directions = [0, 0, 0, 0]
        directions[self.direction] = 1

        state.extend(directions)

        fruit_right = 1 if FRUIT_X > self.head[0] else 0
        fruit_below = 1 if FRUIT_Y > self.head[1] else 0
        fruit_left = 1 if FRUIT_X < self.head[0] else 0
        fruit_above = 1 if FRUIT_Y < self.head[1] else 0

        state.append(fruit_right)
        state.append(fruit_below)
        state.append(fruit_left)
        state.append(fruit_above)

        right = (self.head[0]+1, self.head[1])
        below = (self.head[0], self.head[1]+1)
        left = (self.head[0]-1, self.head[1])
        above = (self.head[0], self.head[1]-1)

        cells = [left, above, right, below]

        for i in range(4):
            if self.check_danger(cells[i][0], cells[i][1]):
                state.append(1)
            else:
                state.append(0)

        return np.asarray(state)

    def state_reward(self):
        return self.get_state(), self.reward

    def next_pos(self):
        """
        Define where the head of the snake is going to be in the next frame,
        given the direction of the snake.
        """

        if self.direction == 0:
            return (self.head[0]+1, self.head[1])
        if self.direction == 1:
            return (self.head[0], self.head[1]+1)
        if self.direction == 2:
            return (self.head[0]-1, self.head[1])
        if self.direction == 3:
            return (self.head[0], self.head[1]-1)

    def check_danger(self, x, y):
        if x < 0 or y < 0 or x >= W or y >= H:
            return True
        for body_part in self.body[0:-1]:
            if body_part.get_coords() == (x, y):
                return True
        # in case the snake just ate (the tail will stay in place)
        if self.body[-1].get_coords() == (x, y) and self.just_ate:
            return True

    def key(self, event):
        if event.keycode == 37:
            self.direction = 2
            self.step(1)
        elif event.keycode == 38:
            self.direction = 3
            self.step(1)
        elif event.keycode == 39:
            self.direction = 0
            self.step(1)
        elif event.keycode == 40:
            self.direction = 1
            self.step(1)

    def step(self, next_turn):
        if self.dead:
            return

        if next_turn == 0:
            # turn right
            self.direction = (self.direction + 1) % 4
        elif next_turn == 1:
            # go straight
            pass
        else:
            self.direction = (self.direction - 1) % 4

        new_head = self.next_pos()

        if self.check_danger(new_head[0], new_head[1]):
            self.died()
            return

        # Make the head become part of the body
        self.head = new_head

        if self.just_ate:
            # If the snake has just eaten, we create a new head to make it grow
            for body_part in self.body:
                body_part.Id += 1

            self.body[0].change_color()
            self.size += 1

            self.body.insert(0, Body(self.head[0], self.head[1], 0,
                                     self.canvas))

            # In case the new fruit just spawned where we are moving
            if FRUIT_X == self.head[0] and FRUIT_Y == self.head[1]:
                if self.canvas is not None:
                    self.canvas.delete(FRUIT)

                spawn_fruit(self, self.canvas)
                self.reward = self.EAT_REWARD

            else:
                self.just_ate = False
                self.reward = self.STEP_PUNISH

        else:
            new_body = self.body[:-1]

            for body_part in self.body[:-1]:
                body_part.Id += 1

            self.body[0].change_color()
            self.body[-1].change_coords(self.head[0], self.head[1])
            self.body[-1].make_head()
            new_body.insert(0, self.body[-1])

            self.body = new_body

            if FRUIT_X == self.head[0] and FRUIT_Y == self.head[1]:
                self.just_ate = True
                self.score += 1

                if self.canvas is not None:
                    self.canvas.delete(FRUIT)

                spawn_fruit(self, self.canvas)
                self.reward = self.EAT_REWARD
            else:
                self.reward = self.STEP_PUNISH

    def get_pos(self, x, y):
        """
        Method to check if there is something at a specific position x,y.
        It will either return "n" (for nothing), "s" (for snake), "f" for
        fruit, "w" (for wall).
        """
        if (x, y) in self.body or [x, y] == self.head:
            return "s"
        elif FRUIT_X == x and FRUIT_Y == y:
            return "f"
        elif x <= 0 or x >= W or y <= 0 or y >= H:
            return "w"
        else:
            return "n"

    def dist_from_fruit(self):
        dist = abs(FRUIT_X - self.head[0])
        dist += abs(FRUIT_Y - self.head[1])
        return dist

    def died(self):
        self.dead = True
        self.reward = self.DEATH_PUNISH

        for body_part in self.body:
            body_part.died()


class Body(object):
    def __init__(self, x, y, Id, canvas):
        self.x = x
        self.y = y
        self.Id = Id
        self.canvas = canvas
        self.draw()

    def change_color(self):
        if self.canvas is not None:
            if self.is_head():
                self.canvas.itemconfigure(self.obj, fill="blue")
            else:
                self.canvas.itemconfigure(self.obj, fill="dodger blue")

    def draw(self):
        if self.is_head():
            fill_color = "blue"
        else:
            fill_color = "dodger blue"

        grid_x = to_grid(self.x)
        grid_y = to_grid(self.y)

        if self.canvas is not None:
            self.obj = self.canvas.create_rectangle(grid_x, grid_y,
                                                    grid_x + GRID_SIZE,
                                                    grid_y + GRID_SIZE,
                                                    fill=fill_color)

    def is_head(self):
        return self.Id == 0

    def make_head(self):
        self.Id = 0
        if self.canvas is not None:
            self.canvas.itemconfigure(self.obj, fill="blue")

    def change_coords(self, x, y):
        self.x = x
        self.y = y
        grid_x = to_grid(self.x)
        grid_y = to_grid(self.y)

        if self.canvas is not None:
            self.canvas.coords(self.obj, grid_x, grid_y, grid_x + GRID_SIZE,
                               grid_y + GRID_SIZE)

    def get_coords(self):
        return (self.x, self.y)

    def died(self):
        if self.canvas is not None:
            self.canvas.itemconfigure(self.obj, fill="white")


if __name__ == "__main__":
    run(display=False, epsilon=1.004, alpha=1e-4, discount=0.8, eps_end=0.004,
        batch_size=8, nb_games=20000, eat=1, death=-1, step=0,
        optim=torch.optim.Adam, loss_fn=torch.nn.MSELoss)
    # run_user()
