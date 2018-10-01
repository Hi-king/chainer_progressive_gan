import chainer
import chainer_progressive_gan


def load_models(resize, use_latent: bool, pooling_comp: float = 1.0, input_channel: int = 3, gpu=-1,
                pretrained_generator="",
                pretrained_discriminator="", pretrained_vectorizer=""):
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()

    if resize == 32:
        channel_evolution = (512, 512, 512, 256)
    elif resize == 128:
        channel_evolution = (512, 512, 512, 512, 256, 128)
    elif resize == 256:
        channel_evolution = (512, 512, 512, 256, 128, 64, 32)
    elif resize == 512:
        channel_evolution = (512, 512, 512, 512, 256, 128, 64, 32)
    elif resize == 1024:
        channel_evolution = (512, 512, 512, 512, 256, 128, 64, 32, 16)
    else:
        raise Exception()

    generator = chainer_progressive_gan.models.progressive_generator.ProgressiveGenerator(
        channel_evolution=channel_evolution, conditional=True)
    generator_smooth = chainer_progressive_gan.models.progressive_generator.ProgressiveGenerator(
        channel_evolution=channel_evolution, conditional=True)
    discriminator = chainer_progressive_gan.models.ProgressiveDiscriminator(
        pooling_comp=pooling_comp, channel_evolution=channel_evolution, first_channel=input_channel + 3)
    vectorizer = chainer_progressive_gan.models.ProgressiveVectorizer(
        pooling_comp=pooling_comp, channel_evolution=channel_evolution, first_channel=input_channel,
        use_both_conditional_and_latent=use_latent)
    if pretrained_generator != "":
        chainer.serializers.load_npz(pretrained_generator, generator)
    if pretrained_discriminator != "":
        chainer.serializers.load_npz(pretrained_discriminator, discriminator)
    if pretrained_vectorizer != "":
        chainer.serializers.load_npz(pretrained_vectorizer, vectorizer)

    # select GPU
    if gpu >= 0:
        generator.to_gpu()
        generator_smooth.to_gpu()
        discriminator.to_gpu()
        vectorizer.to_gpu()
        print("use gpu {}".format(gpu))
    return generator, generator_smooth, discriminator, vectorizer
