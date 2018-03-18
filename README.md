# chainer_progressive_gan
Learning animeface latent space using Progressive GAN

## RUN

```
pip install git+https://github.com/Hi-king/chainer.git@feature/for_gan
python train.py --gpu=1 --resize 256 /mnt/dwango/ogaki/dataset/celeba_faces_kawaii_creator
```

## Result

Stage-by-Stage animation

![](https://raw.githubusercontent.com/Hi-king/chainer_progressive_gan/master/sample/preview.gif)

128x128 sample

![](https://raw.githubusercontent.com/Hi-king/chainer_progressive_gan/master/sample/preview.png)

## Acknowledgements

* Progressive Growing of GANs for Improved Quality, Stability, and Variation
  * Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
  * http://research.nvidia.com/publication/2017-10_Progressive-Growing-of
* chainer-progressive-gan
  * https://github.com/pfnet-research/chainer-gan-lib
  * My implementation highly depends on this repository
* labpcascade_animeface
  * https://github.com/nagadomi/lbpcascade_animeface
