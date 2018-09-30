import pathlib
import sys
from pathlib import Path
from typing import List

import cv2
import numpy
from chainer.dataset import dataset_mixin

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parent.parent))


class InpaintingDatast(dataset_mixin.DatasetMixin):
    def __init__(self, paths: List[Path], resize=None, with_opening=True):
        self.with_opening = with_opening
        self.resize = resize
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def _todata(self, original):
        return (numpy.float32(original).transpose((2, 0, 1)) - 127.5) / 127.5

    def collapse(self, image):
        width, height, _ = image.shape
        collapsed = image.copy()
        for _ in range(numpy.random.randint(0, 3)):
            cv2.ellipse(collapsed,
                        center=(numpy.random.randint(0, width), numpy.random.randint(0, height)),
                        axes=(numpy.random.randint(0, width // 2), numpy.random.randint(0, height // 2)),
                        angle=numpy.random.randint(0, 360),
                        startAngle=0, endAngle=360, thickness=-1, color=(0, 0, 0))
        return collapsed

    def get_example(self, i) -> (numpy.ndarray, numpy.ndarray):
        image_path = self.paths[i]
        image = cv2.imread(str(image_path))
        if not self.resize is None:
            image = cv2.resize(image, self.resize)
        return self._todata(self.collapse(image)), self._todata(image)
