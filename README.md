# Game Image Generation
With this project, you can train a Generative Adversarial Network.  While you can train with any time of image, 
this repository focuses on generating images from games.

## Requirements
- Python 3.6+
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

- DCGAN
- [Biggan](https://arxiv.org/abs/1809.11096)
- [Deep Biggan](https://arxiv.org/abs/1809.11096)

## Configuration File

Starter model configuration files and configuration README can be found in the _configs_ directory.

### Trained Models

[**models/Deep-biggan-bs64-ch128-mxp-n64-trunc0.75**](models/Deep-biggan-bs64-ch128-mxp-n64-trunc0.75)

**Data**
- 194,460 images (84 GB)
- 20 games
- Resolution: 128 x 96

<table>
  <thead><th colspan="3">Training Batch</th></thead>
  <td colspan="3" align="center"><img src="models/Deep-biggan-bs64-ch128-mxp-n64-trunc0.75/images/train_batch.png" alt="Images of training batch"></td>
  <thead> <th colspan="3"> Generated Images </th> </thead>
<td colspan="3" align="center"><img src="models/Deep-biggan-bs64-ch128-mxp-n64-trunc0.75/images/generated_image_68000.png" alt="Images of training batch"></td>
</table>
