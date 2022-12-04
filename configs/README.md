
## Configuration File

These configuration files generate _128x96_ images and are small versions of the models.

### Parameters

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


$image Width = base Width * 2 ^ {upsample Layers}$

$image Height = base Height * 2 ^ {upsample Layers}$


#### Machine-specific parameters

| Name            | Descriptions                                 |
|-----------------|----------------------------------------------|
| GPU count       | amount of GPUs to use, use 0 to use CPU only |



#### Model Architecture
| Name          | Descriptions                                                                                                                                                                                                                                                    |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Type of model | possible options are [_deep-biggan_, _biggan_, _dcgan_].  Default parameters for these models will automatically be loaded if not specified.  Default configurations for each model can be found in the models directory _src/models/[model_type]/defaults.ini_ |

