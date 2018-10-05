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


### Re-training

```
python train.py --gpu=1 --resize 256 your/dataset/path
```

### Result

Stage-by-Stage animation

![](https://raw.githubusercontent.com/Hi-king/chainer_progressive_gan/master/sample/preview.gif)


## Conditional Image Generation

### sketch2img

```
python tools/conditionals/edge2img/predict_edge2img.py --input_image signico_face.png --vectorizer result/edge2img_resize256_stage0.0_batch16_stginterval500000_latentON_1538310505/vectorizer_280000.npz --generator result/edge2img_resize256_stage0.0_batch16_stginterval500000_latentON_1538310505/generator_280000.npz --stage 8 --to_line --use_latent
```

|input |colorized|
|---|---|
|![](./sample/input5.png)|![](./sample/color5_use_latent.png)|

### pose2img

|imageA|poseA|poseB|imageB(generated)|
|---|---|---|---|
|![](./sample/pose2img/input_image.png)|![](./sample/pose2img/input_pose.png)|![](./sample/pose2img/target_pose.png)|![](./sample/pose2img/result_image.png)|


## Acknowledgements

* Progressive Growing of GANs for Improved Quality, Stability, and Variation
  * Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
  * http://research.nvidia.com/publication/2017-10_Progressive-Growing-of
* chainer-gan-lib
  * https://github.com/pfnet-research/chainer-gan-lib
  * My implementation highly depends on this repository
* labpcascade_animeface
  * https://github.com/nagadomi/lbpcascade_animeface
