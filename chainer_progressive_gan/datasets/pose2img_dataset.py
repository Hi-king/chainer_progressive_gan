import glob
import os
import collections
import random

import cv2
import numpy
import tqdm
from PIL import Image
from chainer.dataset import dataset_mixin


class Market1501Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, data_dir, resize):
        """
        :param data_dir: /mnt/dataset/Market-1501/Market-1501-v15.09.15/gt_bbox
        """
        self.resize = resize
        self.pose_dir = data_dir + "_openpose"
        pathdict = collections.defaultdict(list)
        for path in tqdm.tqdm(list(glob.glob("{}/*.jpg".format(data_dir)))):
            basename = os.path.basename(path)
            uid = int(basename.split("_")[0])
            pathdict[uid].append(path)
        self.individuals = list(pathdict.values())

    def __len__(self):
        return len(self.individuals)

    def get_example(self, i):
        list = self.individuals[i]
        random.shuffle(list)
        x, y = list[:2]
        x_item, y_item = self.get_img_pose(x), self.get_img_pose(y)

        input_data = numpy.concatenate((x_item[0], x_item[1], y_item[1]), axis=2)  # image, pose, pose
        target_data = y_item[0]

        # return target_data.transpose(2, 0, 1), input_data.transpose(2, 0, 1)
        if self.resize is None:
            return self._todata(input_data), self._todata(target_data)
        else:
            return self._todata(cv2.resize(input_data, self.resize)), self._todata(cv2.resize(target_data, self.resize))

    @staticmethod
    def read_image(path):
        return cv2.imread(path)[:, :, :3]
        # return numpy.asarray(Image.open(path), dtype=numpy.float32)[:, :, :3]

    @staticmethod
    def resize_pose_with_reference(image, pose, need_shift=False):
        h, w = image.shape[:2]
        rh, rw = pose.shape[:2]

        resized_pose = cv2.resize(pose, (int(rw * float(h) / rh), h))

        if need_shift:
            horizontal = resized_pose.sum(axis=(0, 2))
            if len(horizontal.nonzero()[0]) == 0:
                h_start = 0
            else:
                h_min, h_max = horizontal.nonzero()[0].min(), horizontal.nonzero()[0].max()
                h_start = min(
                    max((h_min + h_max) // 2 - w // 2, 0),
                    resized_pose.shape[1] - w
                )
        else:
            h_start = 0

        return resized_pose[:, h_start:h_start + w, :]

    def get_img_pose(self, image_path):
        basename = os.path.basename(image_path)
        pose_path = os.path.join(
            self.pose_dir,
            os.path.splitext(basename)[0] + "_rendered.png")

        image = self.read_image(image_path)
        pose = self.read_image(pose_path)
        pose = self.resize_pose_with_reference(image, pose)
        return image, pose

    def _todata(self, image):
        return (image.transpose(2, 0, 1).astype(numpy.float32) - 128) / 128
