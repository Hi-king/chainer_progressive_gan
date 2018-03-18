# -*- coding: utf-8 -*-
import chainer
from chainer_gan_lib.progressive.net import EqualizedConv2d, DiscriminatorBlock, EqualizedLinear, minibatch_std


class Vectorizer(chainer.Chain):
    def __init__(self, ch=512, pooling_comp=1.0,
                 channel_evolution=(512, 512, 512, 512, 256, 128, 64, 32, 16)):
        super().__init__()
        self.max_stage = (len(channel_evolution) - 1) * 2
        self.pooling_comp = pooling_comp  # compensation of ave_pool is 0.5-Lipshitz
        with self.init_scope():
            bs = [
                chainer.Link()  # dummy
            ]
            self.fromRGB = EqualizedConv2d(3, channel_evolution[-1], 1, 1, 0)

            for i in range(1, len(channel_evolution)):
                bs.append(DiscriminatorBlock(channel_evolution[i], channel_evolution[i - 1], pooling_comp))
            self.bs = chainer.ChainList(*bs)

            self.out0 = EqualizedConv2d(ch + 1, ch, 3, 1, 1)
            self.out1 = EqualizedConv2d(ch, ch, 4, 1, 0)

    def __call__(self, x):
        stage = self.max_stage
        h = chainer.functions.leaky_relu(self.fromRGB(x))

        for i in range(int(stage // 2), 0, -1):
            h = self.bs[i](h)

        h = minibatch_std(h)
        h = chainer.functions.leaky_relu((self.out0(h)))
        return self.out1(h)
