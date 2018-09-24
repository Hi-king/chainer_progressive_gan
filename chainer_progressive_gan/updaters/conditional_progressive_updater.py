# -*- coding: utf-8 -*-
import chainer
import math
import numpy
import chainer_gan_lib.common.misc


class ConditionalProgressiveUpdater(chainer.training.StandardUpdater):
    def __init__(self, resolution, *args, **kwargs):
        self.resolution = resolution
        self.vec, self.gen, self.dis, self.gs = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.lam = kwargs.pop('lam')
        self.gamma = kwargs.pop('gamma')
        self.smoothing = kwargs.pop('smoothing')
        self.stage_interval = kwargs.pop('stage_interval')
        self.initial_stage = kwargs.pop('initial_stage')
        self.counter = math.ceil(self.initial_stage * self.stage_interval)
        super().__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        vec_optimizer = self.get_optimizer('opt_vec')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            x_real = chainer.Variable(xp.asarray([pair[1] for pair in batch]))
            cond = chainer.Variable(xp.asarray([pair[0] for pair in batch]))

            self.stage = self.counter / self.stage_interval

            if math.floor(self.stage) % 2 == 0:
                reso = min(self.resolution, 4 * 2 ** (((math.floor(self.stage) + 1) // 2)))
                scale = max(1, self.resolution // reso)
                if scale > 1:
                    x_real = chainer.functions.average_pooling_2d(x_real, scale, scale, 0)
                    cond = chainer.functions.average_pooling_2d(cond, scale, scale, 0)
            else:
                alpha = self.stage - math.floor(self.stage)
                reso_low = min(self.resolution, 4 * 2 ** (((math.floor(self.stage)) // 2)))
                reso_high = min(self.resolution, 4 * 2 ** (((math.floor(self.stage) + 1) // 2)))
                scale_low = max(1, self.resolution // reso_low)
                scale_high = max(1, self.resolution // reso_high)
                if scale_low > 1:
                    x_real_low = chainer.functions.unpooling_2d(
                        chainer.functions.average_pooling_2d(x_real, scale_low, scale_low, 0),
                        2, 2, 0, outsize=(reso_high, reso_high))
                    x_real_high = chainer.functions.average_pooling_2d(x_real, scale_high, scale_high, 0)
                    x_real = (1 - alpha) * x_real_low + alpha * x_real_high
                    cond_low = chainer.functions.unpooling_2d(
                        chainer.functions.average_pooling_2d(cond, scale_low, scale_low, 0),
                        2, 2, 0, outsize=(reso_high, reso_high))
                    cond_high = chainer.functions.average_pooling_2d(cond, scale_high, scale_high, 0)
                    cond = (1 - alpha) * cond_low + alpha * cond_high

            y_real = self.dis(chainer.functions.concat([x_real, cond], axis=1), stage=self.stage)

            z = chainer.Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            z_cond = self.vec(cond, stage=self.stage)
            x_fake = self.gen(z, stage=self.stage, skip_hs=z_cond)
            y_fake = self.dis(chainer.functions.concat([x_fake, cond], axis=1), stage=self.stage)

            x_fake.unchain_backward()

            eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            x_mid_v = chainer.Variable(x_mid.data)
            y_mid = chainer.functions.sum(self.dis(chainer.functions.concat([x_mid_v, cond], axis=1), stage=self.stage))

            dydx, = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)
            dydx = chainer.functions.sqrt(chainer.functions.sum(dydx * dydx, axis=(1, 2, 3)))
            loss_gp = self.lam * chainer.functions.mean_squared_error(dydx, self.gamma * xp.ones_like(dydx.data)) * (
                    1.0 / self.gamma ** 2)

            loss_dis = chainer.functions.sum(-y_real) / batchsize
            loss_dis += chainer.functions.sum(y_fake) / batchsize

            # prevent drift factor
            loss_dis += 0.001 * chainer.functions.sum(y_real ** 2) / batchsize

            loss_dis_total = loss_dis + loss_gp
            self.dis.cleargrads()
            loss_dis_total.backward()
            dis_optimizer.update()
            loss_dis_total.unchain_backward()

            # train generator
            z = chainer.Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            z_cond = self.vec(cond, stage=self.stage)
            x_fake = self.gen(z, stage=self.stage, skip_hs=z_cond)
            y_fake = self.dis(chainer.functions.concat([x_fake, cond]), stage=self.stage)
            loss_gen = chainer.functions.sum(-y_fake) / batchsize
            self.gen.cleargrads()
            self.vec.cleargrads()
            loss_gen.backward()
            gen_optimizer.update()
            vec_optimizer.update()

            # update smoothed generator
            chainer_gan_lib.common.misc.soft_copy_param(self.gs, self.gen, 1.0 - self.smoothing)

            chainer.reporter.report({'loss_dis': loss_dis})
            chainer.reporter.report({'loss_gen': loss_gen})
            chainer.reporter.report({'loss_gp': loss_gp})
            chainer.reporter.report({'g': chainer.functions.mean(dydx)})
            chainer.reporter.report({'stage': self.stage})

            self.counter += batchsize
