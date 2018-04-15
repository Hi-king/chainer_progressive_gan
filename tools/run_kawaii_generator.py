import argparse
import sys

import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from chainer_progressive_gan import KawaiiGenerator


def main(args):
    creator = KawaiiGenerator(args.model, args.stage)
    image = creator.create_one()
    image.save("test.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sample/generator_smooth_275000.npz")
    parser.add_argument("--stage", type=int)
    args = parser.parse_args()

    main(args)
