# Game Image Generation
With this project, you can train an Generative Adversarial Network to learn how a game looks like, and create (fake) images that look like the game

## Requirements
- Python 3.6
- Numpy
- [Pytorch](https://github.com/pytorch/pytorch)
- [Torch-Summary](https://github.com/TylerYep/torch-summary)

## Running

Train models using **/src/train_gan.py**

and by modifying the configuration file

From the parent folder, run
```
python3 -m src.train_gan
```

Or

You could run the bash script **train_gan.sh** to start training
```
bash train_gan.sh
```

## Configuration File

Configure model training: **/model_config.ini**

Modifiable parameters are

#### Loading and saving
- Path to training image directory
- Path to directory to save all output

#### Image details

- Size image width, images will be resized to this
- Size image height, images will be resized to this
- Number of channels, black and white (1), or color (3)

#### Machine-specific parameters

- Number of GPUs
- Number of workers

#### Hyper-parameters - 1

- Number of epochs
- Batch size
- Learning rate

#### Hyper-parameters - 2

- Optimizer parameters
- Latent vector size (Given to the generator)
- Size of feature maps in the generator and discriminator

## Images Generated

These models are trained from images that are collected from people speedrunning the game Super Mario 64


### 1st Trained Model

Results are at **/output/Y8Q8**

- 47,000 images (6.6 GB)
- 120 star speedruns of 8 different players
- images are 64 by 64
- 150 epochs

<table>
  <thead><th colspan="3">Training Batch</th></thead>
  <td colspan="3" align="center"><img src="output/Y8Q8/images/training_batch.png" alt="Images of training batch"></td>
  <thead> <th colspan="3"> Generated Images </th> </thead>
  <tr>
      <th>Epoch 50</th>
      <th>Epoch 70</th>
      <th>Epoch 90</th>
  </tr>
  <tr>
      <td align="center"> <img src="output/Y8Q8/images/fake_images_epoch_50.png" alt="Generated images at epoch 50"> </td>
      <td align="center"> <img src="output/Y8Q8/images/fake_images_epoch_70.png" alt="Generated images at epoch 70"> </td>
      <td align="center"> <img src="output/Y8Q8/images/fake_images_epoch_90.png" alt="Generated images at epoch 90"> </td>
  </tr>
</table>

### 2nd Trained Model

Results are at **/output/GC7M**

- 1 million images (279 GB)
- 3 different players (images without star count removed)
- images are 88 by 66
- 8 epochs
- Generator 4.3m trainable parameters, Discriminator 3.8m trainable parameters


<table>
  <thead><th colspan="3">Training Batch</th></thead>
  <td colspan="3" align="center"> <img src="output/GC7M/images/train_batch.png" alt="Images of training batch"> </td>
  <thead> <th colspan="3"> Generated Images </th> </thead>
  <tr>
    <th>Epoch 1</th>
    <th>Epoch 4</th>
    <th>Epoch 8</th>
  </tr>
  <tr>
    <td align="center"> <img src="output/GC7M/images/fake_epoch_0.png" alt="Generated images at epoch 1"> </td>
    <td align="center"> <img src="output/GC7M/images/fake_epoch_3.png" alt="Generated images at epoch 4"> </td>
    <td align="center"> <img src="output/GC7M/images/fake_epoch_7.png" alt="Generated images at epoch 8"> </td>
  </tr>
</table>

### 3rd Trained Model

Results are at **/output/3CK2**

- 1 million images (279 GB)
- 3 different players (images without star count removed)
- images are 88 by 66
- 9 epochs
- Generator 10.6m trainable parameters, Discriminator 5.9m trainable parameters

<table>
  <thead><th colspan="3">Training Batch</th></thead>
  <td colspan="3" align="center"> <img src="output/3CK2/images/train_batch.png" alt="Images of training batch"> </td>
  <thead> <th colspan="3"> Generated Images </th> </thead>
  <tr>
    <th>Epoch 1</th>
    <th>Epoch 5</th>
    <th>Epoch 9</th>
  </tr>
  <tr>
    <td align="center"> <img src="output/3CK2/images/fake_epoch_0.png" alt="Generated images at epoch 1"> </td>
    <td align="center"> <img src="output/3CK2/images/fake_epoch_4.png" alt="Generated images at epoch 5"> </td>
    <td align="center"> <img src="output/3CK2/images/fake_epoch_8.png" alt="Generated images at epoch 9"> </td>
  </tr>
</table>

