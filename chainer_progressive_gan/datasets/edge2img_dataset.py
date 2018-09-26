import pathlib
import sys
from pathlib import Path
from typing import List

import cv2
import numpy
from chainer.dataset import dataset_mixin

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parent.parent))


class Edge2ImgDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths: List[Path], resize=None, with_opening=True):
        self.with_opening = with_opening
        self.resize = resize
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def _todata(self, original):
        return (numpy.float32(original).transpose((2, 0, 1)) - 127.5) / 127.5

    def to_sketch(self, image):
        edge = cv2.Canny(image, 100, 200)
        if self.with_opening:
            kernel = numpy.ones((3, 3), numpy.uint8)
            sketch = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
        else:
            sketch = edge
        return sketch

    def get_example(self, i) -> (numpy.ndarray, numpy.ndarray):
        image_path = self.paths[i]
        image = cv2.imread(str(image_path))
        sketch = self.to_sketch(image)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        if self.resize is None:
            return self._todata(sketch_rgb), self._todata(image)
        else:
            return self._todata(cv2.resize(sketch_rgb, self.resize)), self._todata(cv2.resize(image, self.resize))


class Sketch2ImgDataset(Edge2ImgDataset):
    def to_sketch(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = numpy.ones((3, 3), numpy.uint8)
        return cv2.dilate(gray, kernel=kernel) - gray
