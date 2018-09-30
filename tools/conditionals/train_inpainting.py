import argparse
import pathlib
import sys
import pipe
import glob

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parent.parent.parent))
import chainer_progressive_gan

import train_conditional

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_glob')
    parser.add_argument('--edge', action=argparse._StoreTrueAction)
    train_conditional.shared_args(parser)
    args = parser.parse_args()
    args.prefix = "inpainting"

    paths = glob.glob(args.dataset_glob) | pipe.select(pathlib.Path) | pipe.as_list
    dataset = chainer_progressive_gan.datasets.InpaintingDatast(
        paths, resize=(args.resize, args.resize))
    train_conditional.main(args, dataset)
