# RENJU (5 in a row)

THERE IS A BUG IN MCTS! IN HELPER FUNCTIONS! TODO: FIND AND FIX!
game: "white":"nevanda","result":"black","dump":"h8 g6 h7 h6 j6 g8 g7 f7 j7 k6 j9 f6 j8 j5 j10"}

***

## GUI

RenjuNN plays white here

![RenjuNN plays white](https://github.com/nuwanda57/Renju/blob/master/readme_data/game_won_example.png)

## Requirements

- anaconda (python 3.7)
- keras
- tensorflow
- pygame

Note: if you have issues trying to launch the UI (that uses pygame), that's probably because you are not using anaconda.

## Model Comment

The model is not the best version of itself. To make a better version launch lib/self-train.ipynb, nothing else needed.
Just let it self-train for a week or longer :)

## If you want to play renju

Clone the repository, launch lib/play_renju.py, have fun. (Click on the thumbs-up sticker to start the game. Then choose a color by clicking on it. Now you are all set for the game. When the game is over, you can choose to play another one by clicking on the board one more time.)

## The renju agent

We use two approaches to train the renju agent.
The first one is a supervised technique: we generate training data from a dataset of ~1900000 renju games and train our neural network. The training data can be found [here](https://github.com/dasimagin/renju/tree/master/data). Current agent was trainded only on 10% of the dataset and using number of epochs = 1. Accuracy shown by the agent ~0.45. Training the neural network on the whole dataset is in primary TODOs for this project. Data generating jupyter notebook can be found [here](https://github.com/nuwanda57/Renju/blob/master/lib/build_renju_nn_pretrain_dataset.ipynb), and [here](https://github.com/nuwanda57/Renju/blob/master/renju_nn_pretrain.ipynb) is the trainig notebook itself.

The second one is a reinforcement learning upproach, that uses [MCTS](https://github.com/nuwanda57/Renju/blob/master/lib/MCTS.py). Notebook for self-trainig can be found [here](https://github.com/nuwanda57/Renju/blob/master/lib/self-train.ipynb).


## TODOs

1) Increase the model's quality by training it on a bigger part of the dataset.
2) Increase the model's quality by applying more iteration of a self-play.
3) List of several approaches to consider:
  - use slides to generate new training data (applying for both supervised train and self-train)
  - increase this list
  
## Special Thanks

Special thanks to the project mentor - [Denis Simagin](https://github.com/dasimagin)

