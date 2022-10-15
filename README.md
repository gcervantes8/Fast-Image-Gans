# Game Image Generation
With this project, you can train a Generative Adversarial Network.  While this is a general image GAN framework, this repository focuses on generating fake game images.

## Requirements
- Python 3.6
- [Pytorch](https://pytorch.org/)
- [Pytorch-ignite](https://pytorch.org/ignite/index.html)
- [torchvision](https://pypi.org/project/torchvision/)
- [torchinfo](https://github.com/TylerYep/torchinfo)
- [torch-ema](https://github.com/fadel/pytorch_ema)
- [SciPy](https://scipy.org/install/)

## Running

From the parent folder, you can run this command to start training a DCGAN model
```
python3 -m src.train_gan configs/dcgan_128_96.ini
```

## Models supported


| Model       | Default Configuration File       |
|-------------|----------------------------------|
| DCGAN       | _configs/dcgan_128_96.ini_       |
| Biggan      | _configs/biggan_128_96.ini_      |
| Deep Biggan | _configs/deep_biggan_128_96.ini_ |

The default model configuration files generate images to be _128x96_ and are small version of the models that still
achieve good results with less GPU power as long as you limit the amount of games/classes.

## Configuration File


#### Loading and saving


| Name            | Descriptions                                                   |
|-----------------|----------------------------------------------------------------|
| Model name      | Name of the model to train or inference from                   |
| Model directory | Directory where the models are saved - defaults to **/models** |

#### Data

| Name            | Descriptions                                                                                  |
|-----------------|-----------------------------------------------------------------------------------------------|
| Train Directory | Images to train the model with.  Example data directory can be found in **data/coil-100**     |
| Base width      | width of aspect ratio                                                                         |
| Base height     | height of aspect ratio                                                                        |
| Upsample Layers | Used to get image height and width, adding 1 to upsample layers doubles the width and height. |

```
$image Width = base Width * 2 ^ {upsample Layers}$

$image Height = base Height * 2 ^ {upsample Layers}$
```

#### Machine-specific parameters

| Name            | Descriptions                                 |
|-----------------|----------------------------------------------|
| GPU count       | amount of GPUs to use, use 0 to use CPU only |



#### Model Architecture
| Name          | Descriptions                                                                                                                                                                                                                                                    |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Type of model | possible options are [_deep-biggan_, _biggan_, _dcgan_].  Default parameters for these models will automatically be loaded if not specified.  Default configurations for each model can be found in the models directory _src/models/[model_type]/defaults.ini_ |



## Images Generated

These models are trained from images that are collected from people speedrunning the game Super Mario 64


### 1st Trained Model

More results are at [**models/Y8Q8**](models/Y8Q8)


- 47,000 images (6.6 GB)
- 120 star speedruns of 8 different players
- images are 64 by 64
- 150 epochs

<table>
  <thead><th colspan="3">Training Batch</th></thead>
  <td colspan="3" align="center"><img src="models/Y8Q8/images/training_batch.png" alt="Images of training batch"></td>
  <thead> <th colspan="3"> Generated Images </th> </thead>
  <tr>
      <th>Epoch 50</th>
      <th>Epoch 70</th>
      <th>Epoch 90</th>
  </tr>
  <tr>
      <td align="center"> <img src="models/Y8Q8/images/fake_images_epoch_50.png" alt="Generated images at epoch 50"> </td>
      <td align="center"> <img src="models/Y8Q8/images/fake_images_epoch_70.png" alt="Generated images at epoch 70"> </td>
      <td align="center"> <img src="models/Y8Q8/images/fake_images_epoch_90.png" alt="Generated images at epoch 90"> </td>
  </tr>
</table>

### 2nd Trained Model

More results are at [**models/GC7M**](models/GC7M)


- 1 million images (279 GB)
- 3 different players (images without star count removed)
- images are 88 by 66
- 8 epochs
- Generator 4.3m trainable parameters, Discriminator 3.8m trainable parameters


<table>
  <thead><th colspan="3">Training Batch</th></thead>
  <td colspan="3" align="center"> <img src="models/GC7M/images/train_batch.png" alt="Images of training batch"> </td>
  <thead> <th colspan="3"> Generated Images </th> </thead>
  <tr>
    <th>Epoch 1</th>
    <th>Epoch 4</th>
    <th>Epoch 8</th>
  </tr>
  <tr>
    <td align="center"> <img src="models/GC7M/images/fake_epoch_0.png" alt="Generated images at epoch 1"> </td>
    <td align="center"> <img src="models/GC7M/images/fake_epoch_3.png" alt="Generated images at epoch 4"> </td>
    <td align="center"> <img src="models/GC7M/images/fake_epoch_7.png" alt="Generated images at epoch 8"> </td>
  </tr>
</table>

### 3rd Trained Model

More results are at [**models/3CK2**](models/3CK2)

- 1 million images (279 GB)
- 3 different players (images without star count removed)
- images are 88 by 66
- 9 epochs
- Generator 10.6m trainable parameters, Discriminator 5.9m trainable parameters

<table>
  <thead><th colspan="3">Training Batch</th></thead>
  <td colspan="3" align="center"> <img src="models/3CK2/images/train_batch.png" alt="Images of training batch"> </td>
  <thead> <th colspan="3"> Generated Images </th> </thead>
  <tr>
    <th>Epoch 1</th>
    <th>Epoch 5</th>
    <th>Epoch 9</th>
  </tr>
  <tr>
    <td align="center"> <img src="models/3CK2/images/fake_epoch_0.png" alt="Generated images at epoch 1"> </td>
    <td align="center"> <img src="models/3CK2/images/fake_epoch_4.png" alt="Generated images at epoch 5"> </td>
    <td align="center"> <img src="models/3CK2/images/fake_epoch_8.png" alt="Generated images at epoch 9"> </td>
  </tr>
</table>

### 4th Trained Model

More results are at [**models/KDTI**](models/KDTI)

- 1 million images (279 GB)
- 3 different players (images without star count removed)
- images are 128 by 96
- 7 epochs
- Generator 3.4m trainable parameters, Discriminator 2.8m trainable parameters

<table>
  <thead><th colspan="3">Training Batch</th></thead>
  <td colspan="3" align="center"> <img src="models/KDTI/images/train_batch.png" alt="Images of training batch"> </td>
  <thead> <th colspan="3"> Generated Images </th> </thead>
  <tr>
    <th>Epoch 1</th>
    <th>Epoch 5</th>
    <th>Epoch 7</th>
  </tr>
  <tr>
    <td align="center"> <img src="models/KDTI/images/fake_epoch_1.png" alt="Generated images at epoch 1"> </td>
    <td align="center"> <img src="models/KDTI/images/fake_epoch_5.png" alt="Generated images at epoch 5"> </td>
    <td align="center"> <img src="models/KDTI/images/fake_epoch_7.png" alt="Generated images at epoch 7"> </td>
  </tr>
</table>

