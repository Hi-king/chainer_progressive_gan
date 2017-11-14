import argparse
import glob
import os
import time

import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable
from chainer import training
from chainer.training import extension
from chainer.training import extensions

import datasets
import progressive_updater
from chainer_gan_lib.common.misc import copy_param
from chainer_gan_lib.common.record import record_setting
from chainer_gan_lib.progressive.evaluation import sample_generate, sample_generate_light, calc_inception, calc_FID
from chainer_gan_lib.progressive.net import Discriminator, Generator

import models.progressive_discriminator


def check_chainer_version():
    try:
        x = Variable(np.asarray([1, 2, 3], dtype="f"))
        y = F.sum(1.0 / x)
        y.backward(enable_double_backprop=True, retain_grad=True)
        (F.sum(x.grad_var)).backward()
    except:
        print("This code uses double-bp of DivFromConstant (not yet merged).")
        print("Please merge this PR: https://github.com/chainer/chainer/pull/3615 to chainer.")
        print("    (in chainer repository)")
        print("    git fetch origin pull/3615/head:rdiv")
        print("    git merge rdiv")
        print("    (reinstall chainer")
        exit(0)
    try:
        x = Variable(np.asarray([1, 2, 3], dtype="f"))
        y = F.sum(F.sqrt(x))
        y.backward(enable_double_backprop=True, retain_grad=True)
        (F.sum(x.grad_var)).backward()
    except:
        print("Should use current version of chainer.")
        exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Train script')
    parser.add_argument('dataset_directory')
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    parser.add_argument('--max_iter', '-m', type=int, default=400000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o',
                        help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=25000,
                        help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=50000,
                        help='Interval of evaluation')
    parser.add_argument('--out_image_interval', type=int, default=12500,
                        help='Interval of evaluation')
    parser.add_argument('--stage_interval', type=int, default=400000,
                        help='Interval of stage progress')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--n_dis', type=int, default=1,
                        help='number of discriminator update per generator update')
    parser.add_argument('--lam', type=float, default=10,
                        help='gradient penalty')
    parser.add_argument('--gamma', type=float, default=750,
                        help='gradient penalty')
    parser.add_argument('--pooling_comp', type=float, default=1.0,
                        help='compensation')
    parser.add_argument('--pretrained_generator', type=str, default="")
    parser.add_argument('--pretrained_discriminator', type=str, default="")
    parser.add_argument('--initial_stage', type=float, default=0.0)
    parser.add_argument('--generator_smoothing', type=float, default=0.999)
    args = parser.parse_args()

    result_directory = args.out
    if args.out is None:
        result_directory_name = "_".join([
            "resize{}".format(args.resize),
            str(int(time.time())),
        ])
        result_directory = os.path.join('result', result_directory_name)

    record_setting(result_directory)
    check_chainer_version()

    report_keys = ["stage", "loss_dis", "loss_gp", "loss_gen", "g", "inception_mean", "inception_std", "FID"]
    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    generator = Generator()
    generator_smooth = Generator()
    # discriminator = Discriminator(pooling_comp=args.pooling_comp)
    discriminator = models.progressive_discriminator.ProgressiveDiscriminator(
        pooling_comp=args.pooling_comp, channel_evolution=(512, 512, 512, 256))

    # select GPU
    if args.gpu >= 0:
        generator.to_gpu()
        generator_smooth.to_gpu()
        discriminator.to_gpu()
        print("use gpu {}".format(args.gpu))

    if args.pretrained_generator != "":
        chainer.serializers.load_npz(args.pretrained_generator, generator)
    if args.pretrained_discriminator != "":
        chainer.serializers.load_npz(args.pretrained_discriminator, discriminator)
    copy_param(generator_smooth, generator)

    # Setup an optimizer
    def make_optimizer(model, alpha=0.001, beta1=0.0, beta2=0.99):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(generator)
    opt_dis = make_optimizer(discriminator)

    if args.dataset_directory == 'cifar10':
        import chainer_gan_lib.common.dataset
        train_dataset = chainer_gan_lib.common.dataset.Cifar10Dataset()
    else:
        dataset_pathes = list(glob.glob("{}/*".format(args.dataset_directory)))
        print("use {} files".format(len(dataset_pathes)))
        train_dataset = datasets.ResizedImageDataset(dataset_pathes, resize=(args.resize, args.resize))
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)

    # Set up a trainer
    updater = progressive_updater.ProgressiveUpdater(
        resolution=args.resize,
        models=(generator, discriminator, generator_smooth),
        iterator={
            'main': train_iter},
        optimizer={
            'opt_gen': opt_gen,
            'opt_dis': opt_dis},
        device=args.gpu,
        n_dis=args.n_dis,
        lam=args.lam,
        gamma=args.gamma,
        smoothing=args.generator_smoothing,
        initial_stage=args.initial_stage,
        stage_interval=args.stage_interval
    )
    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=result_directory)
    trainer.extend(extensions.snapshot_object(
        generator, 'generator_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        generator_smooth, 'generator_smooth_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        discriminator, 'discriminator_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))

    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    trainer.extend(sample_generate(generator_smooth, result_directory),
                   trigger=(args.out_image_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(sample_generate_light(generator_smooth, result_directory),
                   trigger=(args.evaluation_interval // 10, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    # trainer.extend(calc_inception(generator_smooth), trigger=(args.evaluation_interval, 'iteration'),
    #                priority=extension.PRIORITY_WRITER)
    # trainer.extend(calc_FID(generator_smooth), trigger=(args.evaluation_interval, 'iteration'),
    #                priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
