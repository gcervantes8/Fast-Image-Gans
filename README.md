# Game Images Generation

With this project, you can train an Generative Adversarial Network to learn how a game looks like, and create (fake) images like the game.

## Requirements
- Python 3.6
- Pytorch
- Numpy

## Running

Train models using **/src/train_gan.py**

### Configuration File

Configure model training: **/model_config.ini**

Modifiable is

#### Loading and Saving
- Path to training image directory
- Path to directory to save all output

#### Hyper-parameters - 1

- Number of epochs
- Batch size
- Learning rate
- Optimizer parameters

#### Hyper-parameters - 2

- Size image, number of pixels images will be resized to (square)
- Number of channels, black and white (1), or color (3)
- Latent vector size (Given to the generator)
= Size of feature maps in the generator and discriminator



#### Mario 64

Mario 64 is the first game tested on.  Images are collected from people speed-running the game, specifically getting 120 stars.  

#### 1st Trained model results

