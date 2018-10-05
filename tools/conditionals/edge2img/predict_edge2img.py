import argparse
import pathlib
import sys

import cv2
import numpy

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parents[3]))
import tools


def main(args: argparse.Namespace):
    generator, _, _, vectorizer = tools.load_models(resize=args.resize, use_latent=args.use_latent,
                                                    pretrained_generator=args.generator,
                                                    pretrained_vectorizer=args.vectorizer)
    image = cv2.imread(str(args.input_image))
    if args.to_line:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = numpy.ones((3, 3), numpy.uint8)
        image = cv2.dilate(gray, kernel=kernel) - gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("line.png", image)
    image = cv2.resize(image, (args.resize, args.resize))

    result_image = tools.utils.predict(image, args.resize, args.stage, vectorizer, generator, cols=args.cols,
                                       rows=args.rows)
    cv2.imwrite("result.png", result_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=pathlib.Path, required=True)
    parser.add_argument('--vectorizer', type=pathlib.Path, required=True)
    parser.add_argument('--generator', type=pathlib.Path, required=True)
    parser.add_argument('--use_latent', action=argparse._StoreTrueAction)
    parser.add_argument('--to_line', action=argparse._StoreTrueAction)
    parser.add_argument('--rows', type=int, default=1)
    parser.add_argument('--cols', type=int, default=1)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--stage', type=int, default=1000)
    args = parser.parse_args()
    main(args)
