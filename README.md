# chainer_progressive_gan
Learning animeface latent space using Progressive GAN

## Create Kawaii !

![](https://raw.githubusercontent.com/Hi-king/chainer_progressive_gan/master/sample/preview.png)


```
pip install git+https://github.com/Hi-king/chainer_progressive_gan.git
# need git-lfs(https://github.com/git-lfs/git-lfs/wiki/Installation)
```

```
from chainer_progressive_gan import KawaiiGenerator
creator = KawaiiGenerator()
image = creator.create_one()
image.save("test.png")
```


## Re-training

```
python train.py --gpu=1 --resize 256 your/dataset/path
```

## Result

Stage-by-Stage animation

![](https://raw.githubusercontent.com/Hi-king/chainer_progressive_gan/master/sample/preview.gif)


## Acknowledgements

* Progressive Growing of GANs for Improved Quality, Stability, and Variation
  * Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
  * http://research.nvidia.com/publication/2017-10_Progressive-Growing-of
* chainer-gan-lib
  * https://github.com/pfnet-research/chainer-gan-lib
  * My implementation highly depends on this repository
* labpcascade_animeface
  * https://github.com/nagadomi/lbpcascade_animeface
