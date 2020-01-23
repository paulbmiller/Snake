# Snake

Reinforcement Learning applied to a 20x20 Snake game with Reinforcement Learning.

The file DeepQLearning.py uses ideas and some code from https://www.youtube.com/watch?v=RfNxXlO6BiA for QLearning, butoes not work as anticipated. Is barely able to eat the first fruit consistently (the initial position and direction of the snake is random). Maybe it could work with some tweaking of the optimizer, reward system, neural net model, loss function, learning rate, environment modelling, state definition, etc.

The file RL.py is my own implementation, which simply uses a Neural Network to find the optimal policy i.e. the optimal action from each state. The input of the NN is the state of the game (which I have reduced to 12 input variables) and the output is a vector of 3 actions (the max being the action the NN will take). It learns at the end of each game with discounted rewards.