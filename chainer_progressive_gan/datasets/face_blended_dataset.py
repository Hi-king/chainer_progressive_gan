import json
import pathlib
import cv2
import numpy

import sys
from pathlib import Path
from typing import List

from chainer.dataset import dataset_mixin

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parent.parent))
import chainer_progressive_gan


class FaceBlendedDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths: List[Path], resize=None):
        self.resize = resize
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def _todata(self, original):
        return (numpy.float32(original).transpose((2, 0, 1)) - 127.5) / 127.5

    def get_example(self, i) -> (numpy.ndarray, numpy.ndarray):
        image_path = self.paths[i]
        json_path = image_path.parent / (image_path.stem + ".json")
        json_data = json.load(json_path.open())
        face_meta = chainer_progressive_gan.models.face_extractor.FaceMeta.from_json(json_data)

        image = cv2.imread(str(image_path))
        original_image = image.copy()
        margin = 0.25
        image[face_meta.y:face_meta.y + face_meta.height, face_meta.x:face_meta.x + face_meta.width] = 0

        # subface
        image[
        face_meta.y + int(margin * face_meta.height):
        face_meta.y + face_meta.height - int(margin * face_meta.height),
        face_meta.x + int(margin * face_meta.width):
        face_meta.x + face_meta.width - int(margin * face_meta.height)] = original_image[
                                                                          face_meta.y + int(margin * face_meta.height):
                                                                          face_meta.y + face_meta.height - int(
                                                                              margin * face_meta.height),
                                                                          face_meta.x + int(margin * face_meta.width):
                                                                          face_meta.x + face_meta.width - int(
                                                                              margin * face_meta.height)]
        if self.resize is None:
            return self._todata(image), self._todata(original_image)
        else:
            return self._todata(cv2.resize(image, self.resize)), self._todata(cv2.resize(original_image, self.resize))
