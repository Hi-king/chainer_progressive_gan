import argparse
import cv2
import pathlib
import sys
import pipe
import glob
import chainer
import numpy

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parents[3]))
import chainer_progressive_gan
import tools


def main(args: argparse.Namespace):
    generator, _, _, vectorizer = tools.load_models(resize=args.resize, use_latent=args.use_latent,
                                                    pretrained_generator=args.generator,
                                                    pretrained_vectorizer=args.vectorizer)
    image = cv2.imread(str(args.input_image))
    image = cv2.resize(image, (args.resize, args.resize))

    if args.to_line:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = numpy.ones((3, 3), numpy.uint8)
        image = cv2.dilate(gray, kernel=kernel) - gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = image.astype(numpy.float32).transpose((2, 0, 1))
    image = (image - 127.5) / 127.5
    image_var = chainer.Variable(image[numpy.newaxis, :, :, :])


    current_resize = min(args.resize, 4 * 2 ** (args.stage // 2))
    scale = args.resize // current_resize
    image_var = chainer.functions.average_pooling_2d(image_var, scale, scale, 0)
    image = (image_var.data[0].transpose((1, 2, 0)) * 127.5 + 127.5).astype(numpy.uint8)
    cv2.imwrite("line.png", image)


    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        z = chainer.Variable(numpy.asarray(generator.make_hidden(1)))
        condition = vectorizer(image_var, stage=args.stage)
        result_image_var = generator(z, stage=args.stage, skip_hs=condition)
        result_image = (result_image_var.data[0].transpose((1, 2, 0)) * 127.5 + 127.5)
        result_image = numpy.clip(result_image, 0.0, 255.0).astype(numpy.uint8)
        cv2.imwrite("result.png", result_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=pathlib.Path, required=True)
    parser.add_argument('--vectorizer', type=pathlib.Path, required=True)
    parser.add_argument('--generator', type=pathlib.Path, required=True)
    parser.add_argument('--use_latent', action=argparse._StoreTrueAction)
    parser.add_argument('--to_line', action=argparse._StoreTrueAction)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--stage', type=int, default=1000)
    args = parser.parse_args()
    main(args)
