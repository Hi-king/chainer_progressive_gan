# -*- coding: utf-8 -*-
import chainer
import math
from chainer_gan_lib.progressive.net import EqualizedConv2d, DiscriminatorBlock, EqualizedLinear, minibatch_std


class ProgressiveDiscriminator(chainer.Chain):
    """
    @see https://github.com/pfnet-research/chainer-gan-lib/blob/master/progressive/net.py

    Some modifications from original
    """

    def __init__(self, ch=512, pooling_comp=1.0,
                 channel_evolution=(512, 512, 512, 512, 256, 128, 64, 32, 16)):
        super(ProgressiveDiscriminator, self).__init__()
        self.max_stage = (len(channel_evolution) - 1) * 2
        self.pooling_comp = pooling_comp  # compensation of ave_pool is 0.5-Lipshitz
        with self.init_scope():
            ins = [
                EqualizedConv2d(3, channel_evolution[0], 1, 1, 0)
            ]
            bs = [
                chainer.Link()  # dummy
            ]
            for i in range(1, len(channel_evolution)):
                ins.append(EqualizedConv2d(3, channel_evolution[i], 1, 1, 0))
                bs.append(DiscriminatorBlock(channel_evolution[i], channel_evolution[i - 1], pooling_comp))
            self.ins = chainer.ChainList(*ins)
            self.bs = chainer.ChainList(*bs)

            self.out0 = EqualizedConv2d(ch + 1, ch, 3, 1, 1)
            self.out1 = EqualizedConv2d(ch, ch, 4, 1, 0)
            self.out2 = EqualizedLinear(ch, 1)

    def __call__(self, x, stage):
        # stage0: in0->m_std->out0_0->out0_1->out0_2
        # stage1: (1-a)*(down->in0) + (a)*(in1->b1) ->m_std->out0->out1->out2
        # stage2: in1->b1->m_std->out0_0->out0_1->out0_2
        # stage3: (1-a)*(down->in1) + (a)*(in2->b2) ->b1->m_std->out0->out1->out2
        # stage4: in2->b2->b1->m_std->out0->out1->out2
        # ...

        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if int(stage) % 2 == 0:
            fromRGB = self.ins[stage // 2]
            h = chainer.functions.leaky_relu(fromRGB(x))
        else:
            fromRGB0 = self.ins[stage // 2]
            fromRGB1 = self.ins[stage // 2 + 1]
            b1 = self.bs[int(stage // 2 + 1)]

            h0 = chainer.functions.leaky_relu(
                fromRGB0(self.pooling_comp * chainer.functions.average_pooling_2d(x, 2, 2, 0)))
            h1 = b1(chainer.functions.leaky_relu(fromRGB1(x)))
            h = (1 - alpha) * h0 + alpha * h1

        for i in range(int(stage // 2), 0, -1):
            h = self.bs[i](h)

        h = minibatch_std(h)
        h = chainer.functions.leaky_relu((self.out0(h)))
        h = chainer.functions.leaky_relu((self.out1(h)))
        return self.out2(h)
