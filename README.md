# Game Image Generation
0;256;0c
With this project, you can train an Generative Adversarial Network to learn how a game looks like, and create (fake) images that look like the game

## Requirements
- Python 3.6
- Pytorch
- Numpy

## Running

Train models using **/src/train_gan.py**

and by modifying the configuration file

From the parent folder, run
```
python3 -m src.train_gan
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

## Images Generated with Training Details


#### Super Mario 64

Images are collected from people speedrunning the game

### Trained model results

Results at **/output/Y8Q8**

- 64 by 64 Images
- 150 epochs


Training batch is 6.6 Gigabytes, of 120 star speedruns of 8 different players

| Training batch  |
|---|
| ![Image of a training batch](output/Y8Q8/images/training_batch.png)  |




##### Generated Images
| Epoch 50  | Epoch 70 | Epoch 90|
|---| ---| --- |
| ![Generated images at epoch 50](output/Y8Q8/images/fake_images_epoch_50.png)  | ![Generated images at epoch 70](output/Y8Q8/images/fake_images_epoch_70.png) | ![Generated images at epoch 90](output/Y8Q8/images/fake_images_epoch_90.png) |







