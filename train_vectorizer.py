import os
import sys
import random
import argparse
import chainer.optimizers
import numpy
import matplotlib

matplotlib.use('Agg')
import pylab

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import models

parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("--iter", type=int, default=5000000)
parser.add_argument("--out_dir", default=None)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument('--resize', type=int, default=32)
args = parser.parse_args()

if args.out_dir is None:
    args.out_dir = os.path.dirname(args.model_file)

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    # chainer.Function.type_check_enable = False
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy


def augment(original_img, max_margin=10):
    margin = random.randint(0, max_margin)
    original_width, original_height, _ = original_img.shape
    left = random.randint(0, margin)
    top = random.randint(0, margin)
    cropped_img = original_img[
                  left:left + (original_width - 2 * margin),
                  top:top + (original_height - 2 * margin),
                  ]
    return cv2.resize(cropped_img, (original_width, original_height))

if args.resize == 32:
    channel_evolution = (512, 512, 512, 256)
elif args.resize == 128:
    channel_evolution = (512, 512, 512, 512, 256, 128)
elif args.resize == 256:
    channel_evolution = (512, 512, 512, 512, 256, 128, 64)  # too much memory
    # channel_evolution = (512, 512, 512, 256, 128, 64, 32)
elif args.resize == 512:
    channel_evolution = (512, 512, 512, 512, 256, 128, 64, 32)
elif args.resize == 1024:
    channel_evolution = (512, 512, 512, 512, 256, 128, 64, 32, 16)
else:
    raise Exception()

generator = models.progressive_generator.ProgressiveGenerator(channel_evolution=channel_evolution)
chainer.serializers.load_npz(args.model_file, generator)
optimizer = chainer.optimizers.Adam(alpha=0.001)
vectorizer = models.vectorizer.Vectorizer(channel_evolution=channel_evolution)
if args.gpu >= 0:
    generator.to_gpu()
    vectorizer.to_gpu()
optimizer.setup(vectorizer)

for i in range(args.iter):
    z_data = (xp.random.uniform(-1, 1, (1, 512, 1, 1)).astype(xp.float32))
    vectorizer.cleargrads()
    generator.cleargrads()
    z = chainer.Variable(z_data)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        x = generator(z)

    # augmentation
    if args.gpu >= 0:
        augmented = xp.array([augment(x.data.get()[0].transpose(1, 2, 0)).transpose(2, 0, 1)])
    else:
        augmented = xp.array([augment(x.data[0].transpose(1, 2, 0)).transpose(2, 0, 1)])

    reconstructed = vectorizer(chainer.Variable(augmented))

    loss = chainer.functions.mean_squared_error(
        reconstructed,
        z
    )
    loss.backward()
    optimizer.update()

    if i % 1000 == 0:
        print("i: {}, loss: {}".format(i, loss.data))

    if i % 10000 == 0:
        print(loss.data)
        chainer.serializers.save_npz(os.path.join(args.out_dir, "vectorizer_model_{}".format(i)), vectorizer)

    if i % 10000 == 0:
        def clip_img(x):
            return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


        def save(x, filepath):
            img = ((numpy.vectorize(clip_img)(x[0, :, :, :]) + 1) / 2).transpose(1, 2, 0)
            pylab.imshow(img)
            pylab.axis('off')
            pylab.savefig(filepath)


        reconstructed = vectorizer(x)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            regenerated = generator(reconstructed)

        if args.gpu >= 0:
            save(x.data.get(), os.path.join(args.out_dir, "constructed.png"))
            save(regenerated.data.get(), os.path.join(args.out_dir, "reconstructed.png"))
        else:
            save(x.data, os.path.join(args.out_dir, "constructed.png"))
            save(regenerated.data, os.path.join(args.out_dir, "reconstructed.png"))
