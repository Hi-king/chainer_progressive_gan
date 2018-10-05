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
                                                    pretrained_vectorizer=args.vectorizer, input_channel=9)
    original_image = cv2.imread(str(args.input_image))
    original_size = original_image.shape[:2][::-1]
    image = cv2.resize(original_image, (args.resize, args.resize))
    pose1 = cv2.resize(cv2.imread(str(args.input_pose)), (args.resize, args.resize))
    pose2 = cv2.resize(cv2.imread(str(args.target_pose)), (args.resize, args.resize))

    image = cv2.blur(image, (3, 3))
    input_data = numpy.concatenate((image, pose1, pose2), axis=2)

    result_image = tools.utils.predict(input_data, args.resize, args.stage, vectorizer, generator, cols=args.cols,
                                       rows=args.rows)
    result_image = cv2.resize(result_image, original_size)
    cv2.imwrite("result.png", result_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=pathlib.Path, required=True)
    parser.add_argument('--input_pose', type=pathlib.Path, required=True)
    parser.add_argument('--target_pose', type=pathlib.Path, required=True)
    parser.add_argument('--vectorizer', type=pathlib.Path, required=True)
    parser.add_argument('--generator', type=pathlib.Path, required=True)
    parser.add_argument('--use_latent', action=argparse._StoreTrueAction)
    parser.add_argument('--rows', type=int, default=1)
    parser.add_argument('--cols', type=int, default=1)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--stage', type=int, default=1000)
    args = parser.parse_args()
    main(args)
