# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:02:16 2019

@author: mille
"""

import tkinter as tk
from random import randint
import time
import numpy as np
import NeuralNet
import torch

WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400
GRID_SIZE = 10
H = WINDOW_HEIGHT // GRID_SIZE
W = WINDOW_WIDTH // GRID_SIZE


def to_grid(x):
    return x * GRID_SIZE

def from_grid(x):
    return x // GRID_SIZE

def random_direction():
        return randint(0,2)

def spawn_fruit(snake, canvas):
    global FRUIT_X, FRUIT_Y, FRUIT
    
    while True:
        FRUIT_X = randint(0, W-1)
        FRUIT_Y = randint(0, H-1)
        if (FRUIT_X, FRUIT_Y) != snake.head \
        and (FRUIT_X, FRUIT_Y) not in map(lambda l:l.get_coords(), snake.body):
            break
    
    FRUIT = canvas.create_rectangle(to_grid(FRUIT_X), to_grid(FRUIT_Y),
                                    to_grid(FRUIT_X) + GRID_SIZE,
                                    to_grid(FRUIT_Y) + GRID_SIZE, fill = "red")


def main():
    window = tk.Tk()
    
    window.winfo_toplevel().title("Snake")
    
    can = tk.Canvas(window, width = WINDOW_WIDTH, 
                    height = WINDOW_HEIGHT, highlightthickness=0)
    
    can.config(bg = "black")
    
    can.pack()
    
    s = Snake(can)
    
    spawn_fruit(s, can)
    
    neural_net = NeuralNet.NN()
    
    window.update()
    
    while not s.dead:
        s.step(randint(0,2))
        
        torch_state = torch.from_numpy(s.get_state()).float()
        
        time.sleep(.1)
        window.update()
        
    print("The snake got a score of {}".format(s.score))
        
    window.mainloop()
    
class Snake(object):
    def __init__(self, canvas):
        self.dead = False
        self.just_ate = False        
        self.size = 3   # Size of the body (does not include the head)
        self.canvas = canvas
        self.score = 0
        
        # x and y position of the head
        self.head = (W//2, H//2)
        
        """Choose a random direction in which to start (0 for right, 1 for 
        down, 2 for left and 3 for up and initialize the snake going straight.
        """
        self.direction = randint(0,3)
        
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
        
        directions = [0,0,0,0]
        directions[self.direction] = 1
        
        state.extend(directions)
        
        fruit_right = 1 if FRUIT_X > self.head[0] else 0
        fruit_below = 1 if FRUIT_Y > self.head[1] else 0
        fruit_left = 1 if FRUIT_X < self.head[0] else 0
        fruit_above = 1 if FRUIT_Y < self.head[1] else 0
        
        state.append(fruit_left)
        state.append(fruit_below)
        state.append(fruit_right)
        state.append(fruit_above)
        
        right = [(self.head[0]+1, self.head[1]),(self.head[0]+2, self.head[1])]
        below = [(self.head[0], self.head[1]+1),(self.head[0], self.head[1]+2)]
        left = [(self.head[0]-1, self.head[1]),(self.head[0]-2, self.head[1])]
        above = [(self.head[0], self.head[1]-1),(self.head[0], self.head[1]-2)]
        
        cells = [right, below, left, above]
        popped = (self.direction + 2) % 4
        cells.pop(popped)
        
        if popped == 1:
            cells[0], cells[1], cells[2] = cells[1], cells[2], cells[0]
        elif popped == 2:
            cells[0], cells[1], cells[2] = cells[2], cells[0], cells[1]
            
        for direction in cells:
            if self.check_danger(direction[0][0], direction[0][1]):
                state.append(1)
                state.append(1)
            elif self.check_danger(direction[1][0] , direction[1][1]):
                state.append(0)
                state.append(1)
            else:
                state.append(0)
                state.append(0)
        
        return np.asarray(state)
    
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
        if x <= 0 or y <= 0 or x >= W or y >= H:
            return True
        for body_part in self.body:
            if body_part.get_coords() == (x, y):
                return True
        
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
        
        if self.just_ate:
            # If the snake has just eaten, we create a new head to make it grow
            
            self.body[0].change_color()
            
            for body_part in self.body:
                body_part.Id += 1
            
            self.size += 1
            
            self.head = new_head
            
            self.body.insert(0, Body(self.head[0], self.head[1], 0, 
                                     self.canvas))            
            
            # In case the new fruit just spawned where we are moving
            if FRUIT_X == self.head[0] and FRUIT_Y == self.head[1]:
                self.just_ate = True
                self.canvas.delete(FRUIT)
                spawn_fruit(self, self.canvas)
                
            else:
                self.just_ate = False
            
        else:
            # Make the head become part of the body
            self.head = new_head
            
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
                self.canvas.delete(FRUIT)
                spawn_fruit(self, self.canvas)
    
    def get_pos(self, x, y):
        """Method to check if there is something at a specific position x,y.
        It will either return "n" (for nothing), "s" (for snake), "f" for
        fruit, "w" (for wall).
        """
        if (x, y) in self.body or [x,y] == self.head:
            return "s"
        if FRUIT_X == x and FRUIT_Y == y:
            return "f"
        if x <= 0 or x >= W or y <= 0 or y >= H:
            return "w"
    
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
    
    def died(self):
        self.dead = True
        
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
        
        self.obj = self.canvas.create_rectangle(grid_x, grid_y, 
                                                grid_x + GRID_SIZE,
                                                grid_y + GRID_SIZE, 
                                                fill=fill_color)
            
    def is_head(self):
        return self.Id == 0
    
    def make_head(self):
        self.Id = 0
        self.canvas.itemconfigure(self.obj, fill="dodger blue")
    
    def change_coords(self, x, y):
        self.x = x
        self.y = y
        grid_x = to_grid(self.x)
        grid_y = to_grid(self.y)
        self.canvas.coords(self.obj, grid_x, grid_y, grid_x + GRID_SIZE,
                           grid_y + GRID_SIZE)
        
    def get_coords(self):
        return (self.x, self.y)
    
    def died(self):
        self.canvas.itemconfigure(self.obj, fill="white")
        

if __name__ == "__main__":
    main()
    
