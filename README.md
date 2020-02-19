# Snake

Reinforcement Learning applied to a 20x20 Snake game with Reinforcement Learning.

The file DeepQLearning.py uses ideas and some code from https://www.youtube.com/watch?v=RfNxXlO6BiA for QLearning, but does not work as anticipated. Is barely able to eat the first fruit consistently (the initial position and direction of the snake is random). Maybe it could work with some tweaking of the optimizer, reward system, neural net model, loss function, learning rate, environment modelling, state definition, etc.

The file RL.py is my own implementation, which simply uses a Neural Network to find the optimal policy i.e. the optimal action from each state. The input of the NN is the state of the game (which I have reduced to 12 boolean variables) and the output is a vector of 3 actions (the max being the action the NN will take). It learns at the end of each game with discounted rewards. This model is able to get a mean of about 18-20 points, but is not consistent either since the fruit spawns randomly and the starting direction of the snake is also random. I think the model needs more information about the environment to consistently be better (here it only sees immediate danger and knows its position). We could give it memory of its past moves/turns or give it the ability to "see" more information (i.e. more than one cell around its head).

Update 19/02/2020:
Added 9 more variables between 0 and 1 to extend the state information to know the 3 last left or right runs it did and where on the board (x and y positions), added automatic saving of the model and optimizer state dictionaries for future use and a folder structure for each run. The model seems to converge slower but with more room for improvement.