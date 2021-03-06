# -*- coding: utf-8 -*-
import chainer
import math
import numpy
from chainer_gan_lib.progressive.net import feature_vector_normalization, EqualizedConv2d, EqualizedDeconv2d, \
    GeneratorBlock


class ProgressiveGenerator(chainer.Chain):
    """
    @see https://github.com/pfnet-research/chainer-gan-lib/blob/master/progressive/net.py

    Some modifications from original
    """

    def __init__(self, n_hidden=512, ch=512,
                 channel_evolution=(512, 512, 512, 512, 256, 128, 64, 32, 16), conditional=False):
        super(ProgressiveGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.max_stage = (len(channel_evolution) - 1) * 2
        with self.init_scope():
            self.c0 = EqualizedConv2d(n_hidden, ch, 4, 1, 3)
            if conditional:
                self.c1 = EqualizedConv2d(ch, ch, 3, 1, 1)
            else:
                self.c1 = EqualizedConv2d(ch, ch, 3, 1, 1)
            bs = [
                chainer.Link()  # dummy
            ]
            outs = [

            ]
            if conditional:
                outs.append(EqualizedConv2d(channel_evolution[0], 3, 1, 1, 0))
            else:
                outs.append(EqualizedConv2d(channel_evolution[0], 3, 1, 1, 0))

            for i in range(1, len(channel_evolution)):
                if conditional:
                    bs.append(GeneratorBlock(channel_evolution[i - 1] * 2, channel_evolution[i]))
                    outs.append(EqualizedConv2d(channel_evolution[i], 3, 1, 1, 0))
                else:
                    bs.append(GeneratorBlock(channel_evolution[i - 1], channel_evolution[i]))
                    outs.append(EqualizedConv2d(channel_evolution[i], 3, 1, 1, 0))
            self.bs = chainer.ChainList(*bs)
            self.outs = chainer.ChainList(*outs)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1)) \
            .astype(numpy.float32)
        z /= xp.sqrt(xp.sum(z * z, axis=1, keepdims=True) / self.n_hidden + 1e-8)
        return z

    def __call__(self, z, stage=None, test_resolution=32, skip_hs=None):
        # stage0: c0->c1->out0
        # stage1: c0->c1-> (1-a)*(up->out0) + (a)*(b1->out1)
        # stage2: c0->c1->b1->out1
        # stage3: c0->c1->b1-> (1-a)*(up->out1) + (a)*(b2->out2)
        # stage4: c0->c1->b2->out2
        # ...
        if stage is None:
            stage = self.max_stage
        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        z_shape = (len(z), self.n_hidden, 1, 1)
        if (skip_hs is not None) and (skip_hs[-1].shape == z_shape):
            h = skip_hs.pop(-1)
        else:
            h = chainer.functions.reshape(z, z_shape)
        h = chainer.functions.leaky_relu(feature_vector_normalization(self.c0(h)))
        h = chainer.functions.leaky_relu(feature_vector_normalization(self.c1(h)))

        for i in range(1, int(stage // 2 + 1)):
            if skip_hs is not None:  # conditional
                h = chainer.functions.concat([h, skip_hs[- i]], axis=1)
            h = self.bs[i](h)

        if int(stage) % 2 == 0:
            out = self.outs[int(stage // 2)]
            # print(h.shape)
            x = out(h)
        else:
            out_prev = self.outs[stage // 2]
            out_curr = self.outs[stage // 2 + 1]
            b_curr = self.bs[stage // 2 + 1]

            x_0 = out_prev(chainer.functions.unpooling_2d(h, 2, 2, 0, outsize=(2 * h.shape[2], 2 * h.shape[3])))
            if skip_hs is not None:  # conditional
                skip_hs_original = skip_hs[- int(stage // 2 + 1)]
                h = chainer.functions.concat([h, skip_hs_original], axis=1)
            h = b_curr(h)
            x_1 = out_curr(h)
            x = (1.0 - alpha) * x_0 + alpha * x_1

        return x
