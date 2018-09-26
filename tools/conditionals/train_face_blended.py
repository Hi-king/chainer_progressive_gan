import argparse
import pathlib
import sys

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parent.parent.parent))
import chainer_progressive_gan

import train_conditional

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_directory', type=pathlib.Path)
    train_conditional.shared_args(parser)
    args = parser.parse_args()
    args.prefix = "face_blend"
    dataset = chainer_progressive_gan.datasets.FaceBlendedDataset(
        list(args.dataset_directory.glob("*.png")),
        resize=(args.resize, args.resize), gray=args.gray_condition)
    train_conditional.main(args, dataset)
