import os
import chainer
import numpy
from PIL import Image

import chainer_progressive_gan


def make_image(gen, stage, seed=0, rows=10, cols=10):
    import numpy as np
    from chainer import Variable
    # np.random.seed(seed)
    n_images = rows * cols
    xp = gen.xp
    z = Variable(xp.asarray(gen.make_hidden(n_images)))
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        x = gen(z, stage=stage)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, h, w = x.shape
    x = x.reshape((rows, cols, 3, h, w))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * h, cols * w, 3))

    preview_path = "testsome.png"
    # if not os.path.exists(preview_dir):
    #     os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)


class KawaiiGenerator(object):
    def __init__(self, model=None, stage: int = None):
        if model is None:
            model = os.path.join(os.path.dirname(__file__), "..", "sample", "generator_smooth_275000.npz")
        self.stage = stage
        self.generator = chainer_progressive_gan.models.progressive_generator.ProgressiveGenerator(
            channel_evolution=(512, 512, 512, 512, 256, 128)
        )
        chainer.serializers.load_npz(model, self.generator)

        self.xp = numpy

    @staticmethod
    def to_image_data(x):
        return numpy.asarray(numpy.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=numpy.uint8)[0].transpose(1, 2, 0)

    @staticmethod
    def to_image(x):
        return Image.fromarray(KawaiiGenerator.to_image_data(x))

    def create_one_debug(self):
        make_image(self.generator, rows=1, cols=1, seed=199, stage=self.stage)
        return None

    def create_one(self):
        z_data = (self.xp.random.normal(size=(1, 512, 1, 1)).astype(self.xp.float32))
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            predicted = self.generator(z_data)
            return self.to_image(chainer.cuda.to_cpu(predicted.data))
