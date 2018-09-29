# -*- coding: utf-8 -*-
import chainer
import math
from chainer_gan_lib.progressive.net import EqualizedConv2d, DiscriminatorBlock, EqualizedLinear, minibatch_std


class ProgressiveVectorizer(chainer.Chain):
    """
    @see https://github.com/pfnet-research/chainer-gan-lib/blob/master/progressive/net.py

    Some modifications from original
    """

    def __init__(self, ch=512, pooling_comp=1.0,
                 channel_evolution=(512, 512, 512, 512, 256, 128, 64, 32, 16), conditional=False,
                 use_both_conditional_and_latent=False):
        super().__init__()
        self.use_both_conditional_and_latent = use_both_conditional_and_latent
        self.max_stage = (len(channel_evolution) - 1) * 2
        self.pooling_comp = pooling_comp  # compensation of ave_pool is 0.5-Lipshitz
        first_channel = 6 if conditional else 3
        with self.init_scope():
            ins = [
                EqualizedConv2d(first_channel, channel_evolution[0], 1, 1, 0)
            ]
            bs = [
                chainer.Link()  # dummy
            ]
            for i in range(1, len(channel_evolution)):
                ins.append(EqualizedConv2d(first_channel, channel_evolution[i], 1, 1, 0))
                bs.append(DiscriminatorBlock(channel_evolution[i], channel_evolution[i - 1], pooling_comp))
            self.ins = chainer.ChainList(*ins)
            self.bs = chainer.ChainList(*bs)
            self.out0 = EqualizedConv2d(ch + 1, ch, 3, 1, 1)
            self.out1 = EqualizedConv2d(ch, ch, 4, 1, 0)
            self.out2 = EqualizedLinear(ch, 1)

    def _c(self, x, stage):
        # stage0: in0->m_std->out0_0->out0_1->out0_2
        # stage1: (1-a)*(down->in0) + (a)*(in1->b1) ->m_std->out0->out1->out2
        # stage2: in1->b1->m_std->out0_0->out0_1->out0_2
        # stage3: (1-a)*(down->in1) + (a)*(in2->b2) ->b1->m_std->out0->out1->out2
        # stage4: in2->b2->b1->m_std->out0->out1->out2
        # ...

        # print(stage)
        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        hs = []
        if int(stage) % 2 == 0:
            fromRGB = self.ins[stage // 2]
            h = chainer.functions.leaky_relu(fromRGB(x))
            hs.append(h)
        else:
            fromRGB0 = self.ins[stage // 2]
            fromRGB1 = self.ins[stage // 2 + 1]
            b1 = self.bs[int(stage // 2 + 1)]
            h0 = chainer.functions.leaky_relu(
                fromRGB0(self.pooling_comp * chainer.functions.average_pooling_2d(x, 2, 2, 0)))
            # hs.append(h0)
            h1 = chainer.functions.leaky_relu(fromRGB1(x))
            h1 = b1(h1)
            h = (1 - alpha) * h0 + alpha * h1
            hs.append(h)
        for i in range(int(stage // 2), 0, -1):
            h = self.bs[i](h)
            hs.append(h)
        h = minibatch_std(h)
        h = chainer.functions.leaky_relu((self.out0(h)))
        h = chainer.functions.leaky_relu((self.out1(h)))
        hs.append(h)
        h = self.out2(h)
        hs.append(h)
        return hs

    def __call__(self, x, stage):
        if self.use_both_conditional_and_latent:
            return self._c(x, stage)[:-2]
        else:
            return self._c(x, stage)[:-1]


class ProgressiveDiscriminator(ProgressiveVectorizer):
    def __call__(self, x, stage):
        return super()._c(x, stage)[-1]
