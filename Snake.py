# -*- coding: utf-8 -*-
import tkinter as tk
import numpy as np
import time
from math import sqrt
from random import randint
from DeepQLearning import DeepQNAgent

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


def start_snake(display=False, display_title='Snake'):
    if display:
        window = tk.Tk()

        window.winfo_toplevel().title(display_title)

        can = tk.Canvas(window, width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                        highlightthickness=0)

        can.config(bg="black")

        can.pack()

    else:
        window, can = None, None

    s = Snake(can)

    spawn_fruit(s, can)

    return window, s, can


def run_user():
    window, s, can = start_snake(display=True)
    can.bind("<Key>", s.key)
    can.focus_set()

    try:
        window.mainloop()
    except KeyboardInterrupt:
        if window:
            window.quit()
            window.destroy()


def run(display=False, debug=False, debug_every=10):
    agent = DeepQNAgent(gamma=0.95, epsilon=1.0, alpha=0.003, max_memory=5000,
                        replace=None)

    window = None

    print("Initializing memory")

    while agent.mem_cntr < agent.mem_size:
        window, s, can = start_snake()
        state_old = s.get_state()
        game_ended = False
        while not game_ended and agent.mem_cntr < agent.mem_size:
            action = randint(0, 2)
            s.step(action)
            state_new = s.get_state()
            agent.store_transition(state_old, action, s.reward, state_new)
            state_old = state_new
            if s.dead:
                game_ended = True

    print("Done initializing memory of size {}".format(agent.mem_cntr))

    scores = []
    eps_history = []
    num_games = 10000
    batch_size = 32
    window = None
    debug_cntr = 0

    try:
        for i in range(1, num_games+1):
            game_ended = False
            steps = 0

            str_out = "Starting game {}".format(i)
            if agent.EPSILON == 0:
                print(str_out)
            else:
                str_out += ", epsilon : {:.3f}".format(agent.EPSILON)
                print(str_out)
                
            eps_history.append(agent.EPSILON)

            if agent.EPSILON == 0 and i % debug_every == 0:
                win_title = 'Snake {}'.format(i)
                window, s, can = start_snake(display=True,
                                             display_title=win_title)
                window.update()

            else:
                window, s, can = start_snake(display=False)

            state_old = s.get_state()
            while not game_ended:
                if window is not None:
                    time.sleep(0.05)

                action = agent.choose_action(state_old)
                s.step(action)

                state_new = s.get_state()
                agent.store_transition(state_old, action, s.reward, state_new)

                steps += 1
                if window is not None:
                    window.update()

                state_old = state_new

                if debug_cntr >= debug_every and debug:
                    agent.learn(batch_size, debug=True)
                    debug_cntr = 0
                else:
                    agent.learn(batch_size, debug=False)
                    if debug:
                        debug_cntr += 1

                if s.dead:
                    game_ended = True

            scores.append(s.score)

            if window is not None:
                time.sleep(0.1)
                window.destroy()

            print("Score : {}, steps: {}".format(s.score, steps))

    except KeyboardInterrupt:
        if window is not None:
            window.destroy()


class Snake(object):
    def __init__(self, canvas):
        self.dead = False
        self.just_ate = False
        self.size = 3   # Size of the body (does not include the head)
        self.canvas = canvas
        self.score = 0
        self.reward = 0

        # x and y position of the head
        self.head = (W//2, H//2)

        """Choose a random direction in which to start (0 for right, 1 for
        down, 2 for left and 3 for up and initialize the snake going straight.
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
        below = (self.head[0]+1, self.head[1])
        left = (self.head[0]+1, self.head[1])
        above = (self.head[0]+1, self.head[1])

        cells = [left, above, right, below]

        for i in range(4):
            if self.direction == i:
                pass
            elif self.check_danger(cells[i][0], cells[i][1]):
                state.append(1)
            else:
                state.append(0)

        """
        right = [(self.head[0]+1, self.head[1]),(self.head[0]+2, self.head[1])]
        below = [(self.head[0], self.head[1]+1),(self.head[0], self.head[1]+2)]
        left = [(self.head[0]-1, self.head[1]),(self.head[0]-2, self.head[1])]
        above = [(self.head[0], self.head[1]-1),(self.head[0], self.head[1]-2)]

        cells = [left, above, right, below]

        for i in range(4):
            if self.direction == i:
                state.extend([1,1])
            elif self.check_danger(cells[i][0][0], cells[i][0][1]):
                state.extend([1,1])
            elif self.check_danger(cells[i][1][0], cells[i][1][1]):
                state.extend([0,1])
            else:
                state.extend([0,0])


        popped = (self.direction + 2) % 4
        cells.pop(popped)

        if popped == 1:
            cells[0], cells[1], cells[2] = cells[1], cells[2], cells[0]
        elif popped == 2:
            cells[0], cells[1], cells[2] = cells[2], cells[0], cells[1]

        for direction in cells:
            if self.check_danger(direction[0][0], direction[0][1]):
                state.extend([1,1])
            elif self.check_danger(direction[1][0] , direction[1][1]):
                state.extend([0,1])
            else:
                state.extend([0,0])
        """

        return np.asarray(state)

    def state_reward(self):
        return self.get_state(), self.reward

    def next_pos(self, head_direction):
        """Define where the head of the snake is going to be in the next frame,
        given the direction where the head is going and the general direction
        of the snake."""

        if head_direction == 0:
            self.direction = (self.direction - 1) % 4
        elif head_direction == 2:
            self.direction = (self.direction + 1) % 4

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
            if self.direction == 0:
                pass
            else:
                self.direction = 2
                self.step(1)
        elif event.keycode == 38:
            if self.direction == 1:
                pass
            else:
                self.direction = 3
                self.step(1)

        elif event.keycode == 39:
            if self.direction == 2:
                pass
            else:
                self.direction = 0
                self.step(1)

        elif event.keycode == 40:
            if self.direction == 3:
                pass
            else:
                self.direction = 1
                self.step(1)

    def step(self, head_direction):
        """Move the snake one cell in head_direction (0 for the snake to turn
        left, 1 for the snake to go straight and 2 for going right).
        """

        if self.dead:
            return

        new_head = self.next_pos(head_direction)

        if self.check_danger(new_head[0], new_head[1]):
            self.died()
            return

        dist_fruit0 = self.get_dist_from_fruit()
        # Make the head become part of the body
        self.head = new_head
        dist_fruit1 = self.get_dist_from_fruit()

        if dist_fruit0 > dist_fruit1:
            moved_closer = True
        else:
            moved_closer = False

        if self.just_ate:
            # If the snake has just eaten, we create a new head to make it grow

            self.body[0].change_color()

            for body_part in self.body:
                body_part.Id += 1

            self.size += 1

            self.body.insert(0, Body(self.head[0], self.head[1], 0,
                                     self.canvas))

            # In case the new fruit just spawned where we are moving
            if FRUIT_X == self.head[0] and FRUIT_Y == self.head[1]:
                self.just_ate = True

                if self.canvas is not None:
                    self.canvas.delete(FRUIT)

                spawn_fruit(self, self.canvas)
                self.reward = 100

            else:
                self.just_ate = False
                if moved_closer:
                    self.reward = 0.1
                else:
                    self.reward = -0.1

        else:
            new_body = self.body[:-1]

            self.body[0].change_color()
            for body_part in self.body[:-1]:
                body_part.Id += 1

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
                self.reward = 100

            elif moved_closer:
                self.reward = 0.1

            else:
                self.reward = -0.1

    def get_pos(self, x, y):
        """Method to check if there is something at a specific position x,y.
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

    def change_dir(self, new_dir):
        """This method will change the direction in which the snake is heading
        only if the new direction is not making it go into itself or the
        direction isn't being changed.
        """
        if self.direction == 0 or self.direction == 2:
            if new_dir == 1 or new_dir == 3:
                self.direction = new_dir

        if self.direction == 1 or self.direction == 3:
            if new_dir == 0 or new_dir == 2:
                self.direction = new_dir

    def get_dist_from_fruit(self):
        dist = abs(FRUIT_X - self.head[0])
        dist += abs(FRUIT_Y - self.head[1])
        return dist

    def died(self):
        self.dead = True
        self.reward = -50

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
            self.canvas.itemconfigure(self.obj, fill="dodger blue")

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
    run(display=True, debug=False, debug_every=1)
