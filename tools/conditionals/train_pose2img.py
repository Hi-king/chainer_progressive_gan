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
    train_conditional.shared_args(parser)
    args = parser.parse_args()
    args.prefix = "market1501"
    args.input_channel = 9

    # paths = glob.glob(args.dataset_glob) | pipe.select(pathlib.Path) | pipe.as_list
    dataset = chainer_progressive_gan.datasets.Market1501Dataset(
        data_dir="/mnt/dataset/Market-1501/Market-1501-v15.09.15/gt_bbox", resize=(args.resize, args.resize))
    # dataset = chainer_progressive_gan.datasets.Edge2ImgDataset(
    #     paths, resize=(args.resize, args.resize))
    train_conditional.main(args, dataset)
