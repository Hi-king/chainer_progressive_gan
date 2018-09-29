import pathlib

import chainer
import numpy
import cv2
import random
import math


class GenerateSampleWithCondition(chainer.training.extension.Extension):
    def __init__(self, encoder, generator, input_dataset, output_dir: pathlib.Path, rows=10, cols=10):
        self.output_dir = output_dir
        self.cols = cols
        self.rows = rows
        self.input_dataset = input_dataset
        self.generator = generator
        self.encoder = encoder

    def _resize(self, stage, original):
        cond = original
        resolution = original.shape[-1]
        if math.floor(stage) % 2 == 0:
            reso = min(resolution, 4 * 2 ** (((math.floor(stage) + 1) // 2)))
            scale = max(1, resolution // reso)
            if scale > 1:
                cond = chainer.functions.average_pooling_2d(cond, scale, scale, 0)
        else:
            alpha = stage - math.floor(stage)
            reso_low = min(resolution, 4 * 2 ** (((math.floor(stage)) // 2)))
            reso_high = min(resolution, 4 * 2 ** (((math.floor(stage) + 1) // 2)))
            scale_low = max(1, resolution // reso_low)
            scale_high = max(1, resolution // reso_high)
            if scale_low > 1:
                cond_low = chainer.functions.unpooling_2d(
                    chainer.functions.average_pooling_2d(cond, scale_low, scale_low, 0),
                    2, 2, 0, outsize=(reso_high, reso_high))
                cond_high = chainer.functions.average_pooling_2d(cond, scale_high, scale_high, 0)
                cond = (1 - alpha) * cond_low + alpha * cond_high
        return cond

    def _tiling(self, x:chainer.Variable):
        x = chainer.cuda.to_cpu(x.data)
        x = numpy.asarray(numpy.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=numpy.uint8)
        _, _, h, w = x.shape
        x = x.reshape((self.rows, self.cols, 3, h, w))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((self.rows * h, self.cols * w, 3))
        return x

    def __call__(self, trainer):
        xs = []
        ys = []
        xp = self.generator.xp
        n_images = self.cols * self.rows
        for _ in range(n_images):
            x, y = self.input_dataset[random.randint(0, len(self.input_dataset))]
            xs.append(x)
            ys.append(y)
        x_var = chainer.Variable(xp.array(xs))
        y_var = chainer.Variable(numpy.array(ys))
        x_var = self._resize(trainer.updater.stage, x_var)
        z = chainer.Variable(xp.asarray(self.generator.make_hidden(n_images)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            condition = self.encoder(x_var, stage=trainer.updater.stage)
            x = self.generator(z, stage=trainer.updater.stage, skip_hs=condition)

        if not self.output_dir.exists():
            self.output_dir.mkdir()
        cv2.imwrite(str(self.output_dir / 'image_result_{:0>8}.png'.format(trainer.updater.iteration)), self._tiling(x))
        cv2.imwrite(str(self.output_dir / 'image_condition_{:0>8}.png'.format(trainer.updater.iteration)), self._tiling(x_var))
        cv2.imwrite(str(self.output_dir / 'image_ground_truth_{:0>8}.png'.format(trainer.updater.iteration)), self._tiling(y_var))
