import argparse
import pathlib
import sys
import time
import pipe
import chainer

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parent.parent))
import chainer_progressive_gan
import chainer_gan_lib.common.record
import chainer_gan_lib.common.misc
import tools
import train


def main(args: argparse.Namespace, dataset):
    result_directory_name = "_".join(
        [
            args.prefix,
            "resize{}".format(args.resize),
            "stage{}".format(args.initial_stage),
            "batch{}".format(args.batchsize),
            "stginterval{}".format(args.stage_interval),
            "latent{}".format("ON" if args.use_latent else "OFF"),
            str(int(time.time())),
        ] | pipe.where(lambda x: len(x) > 0))
    result_directory = args.out / result_directory_name
    chainer_gan_lib.common.record.record_setting(str(result_directory))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
    generator, generator_smooth, discriminator, vectorizer = tools.load_models(
        resize=args.resize, use_latent=args.use_latent,
        input_channel=args.input_channel,
        pretrained_generator=args.pretrained_generator,
        pretrained_vectorizer=args.pretrained_vectorizer)

    # if args.resize == 32:
    #     channel_evolution = (512, 512, 512, 256)
    # elif args.resize == 128:
    #     channel_evolution = (512, 512, 512, 512, 256, 128)
    # elif args.resize == 256:
    #     channel_evolution = (512, 512, 512, 256, 128, 64, 32)
    # elif args.resize == 512:
    #     channel_evolution = (512, 512, 512, 512, 256, 128, 64, 32)
    # elif args.resize == 1024:
    #     channel_evolution = (512, 512, 512, 512, 256, 128, 64, 32, 16)
    # else:
    #     raise Exception()
    #
    # generator = chainer_progressive_gan.models.progressive_generator.ProgressiveGenerator(
    #     channel_evolution=channel_evolution, conditional=True)
    # generator_smooth = chainer_progressive_gan.models.progressive_generator.ProgressiveGenerator(
    #     channel_evolution=channel_evolution, conditional=True)
    # discriminator = chainer_progressive_gan.models.ProgressiveDiscriminator(
    #     pooling_comp=args.pooling_comp, channel_evolution=channel_evolution, first_channel=args.input_channel + 3)
    # vectorizer = chainer_progressive_gan.models.ProgressiveVectorizer(
    #     pooling_comp=args.pooling_comp, channel_evolution=channel_evolution, first_channel=args.input_channel,
    #     use_both_conditional_and_latent=args.use_latent)
    train_iter = chainer.iterators.MultithreadIterator(dataset, args.batchsize)

    # select GPU
    if args.gpu >= 0:
        generator.to_gpu()
        generator_smooth.to_gpu()
        discriminator.to_gpu()
        vectorizer.to_gpu()
        print("use gpu {}".format(args.gpu))

    chainer_gan_lib.common.misc.copy_param(generator_smooth, generator)
    opt_gen = train.make_optimizer(generator)
    opt_dis = train.make_optimizer(discriminator)
    opt_vec = train.make_optimizer(vectorizer)

    updater = chainer_progressive_gan.updaters.ConditionalProgressiveUpdater(
        resolution=args.resize,
        models=(vectorizer, generator, discriminator, generator_smooth),
        iterator={
            'main': train_iter},
        optimizer={
            'opt_vec': opt_vec,
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
    report_keys = ["stage", "loss_dis", "loss_gp", "loss_gen", "g", "inception_mean", "inception_std", "FID"]
    trainer = chainer.training.Trainer(updater, (args.max_iter, 'iteration'), out=str(result_directory))
    trainer.extend(chainer.training.extensions.snapshot_object(
        generator, 'generator_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(chainer.training.extensions.snapshot_object(
        generator_smooth, 'generator_smooth_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(chainer.training.extensions.snapshot_object(
        discriminator, 'discriminator_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(chainer.training.extensions.snapshot_object(
        vectorizer, 'vectorizer_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))

    trainer.extend(chainer.training.extensions.LogReport(keys=report_keys,
                                                         trigger=(args.display_interval, 'iteration')))
    trainer.extend(chainer.training.extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    trainer.extend(
        chainer_progressive_gan.training.GenerateSampleWithCondition(vectorizer, generator, input_dataset=dataset,
                                                                     output_dir=result_directory, rows=3, cols=3),
        trigger=(args.out_image_interval, 'iteration'))
    # trainer.extend(sample_generate(generator_smooth, result_directory),
    #                trigger=(args.out_image_interval, 'iteration'),
    #                priority=extension.PRIORITY_WRITER)
    # trainer.extend(sample_generate_light(generator_smooth, result_directory),
    #                trigger=(args.evaluation_interval // 10, 'iteration'),
    #                priority=extension.PRIORITY_WRITER)
    # trainer.extend(calc_inception(generator_smooth), trigger=(args.evaluation_interval, 'iteration'),
    #                priority=extension.PRIORITY_WRITER)
    # trainer.extend(calc_FID(generator_smooth), trigger=(args.evaluation_interval, 'iteration'),
    #                priority=extension.PRIORITY_WRITER)
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))
    trainer.run()


def shared_args(parser):
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--gray_condition', action=argparse._StoreTrueAction)
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    parser.add_argument('--max_iter', '-m', type=int, default=4000000)
    parser.add_argument('--use_latent', action=argparse._StoreTrueAction)

    # optional
    parser.add_argument("--prefix")
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--out', '-o', default="result",
                        help='Directory to output the result', type=pathlib.Path)
    parser.add_argument('--snapshot_interval', type=int, default=5000,
                        help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=50000,
                        help='Interval of evaluation')
    parser.add_argument('--out_image_interval', type=int, default=5000,
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
    parser.add_argument('--generator_smoothing', type=float, default=0.999)
    parser.add_argument('--initial_stage', type=float, default=0.0)
    parser.add_argument('--pretrained_generator', type=str, default="")
    parser.add_argument('--pretrained_discriminator', type=str, default="")
    parser.add_argument('--pooling_comp', type=float, default=1.0,
                        help='compensation')
