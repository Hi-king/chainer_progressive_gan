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
    def __init__(self, paths: List[Path], resize=None, gray=False):
        self.gray = gray
        self.resize = resize
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def to_blend_image(image1, face_meta1, image2, face_meta2, gray=True):
        margin = 0.25
        image1[face_meta1.y:face_meta1.y + face_meta1.height, face_meta1.x:face_meta1.x + face_meta1.width] = 0

        # subface
        target_image = cv2.cvtColor(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if gray else image2
        face_image2 = target_image[
                      face_meta2.y + int(margin * face_meta2.height):
                      face_meta2.y + face_meta2.height - int(margin * face_meta2.height),
                      face_meta2.x + int(margin * face_meta2.width):
                      face_meta2.x + face_meta2.width - int(margin * face_meta2.height)]

        face_image1 = image1[
                      face_meta1.y + int(margin * face_meta1.height):
                      face_meta1.y + face_meta1.height - int(margin * face_meta1.height),
                      face_meta1.x + int(margin * face_meta1.width):
                      face_meta1.x + face_meta1.width - int(margin * face_meta1.height)]
        image1[
        face_meta1.y + int(margin * face_meta1.height):
        face_meta1.y + face_meta1.height - int(margin * face_meta1.height),
        face_meta1.x + int(margin * face_meta1.width):
        face_meta1.x + face_meta1.width - int(margin * face_meta1.height)] = cv2.resize(face_image2,
                                                                                        face_image1.shape[:2])
        return image1

    def _todata(self, original):
        return (numpy.float32(original).transpose((2, 0, 1)) - 127.5) / 127.5

    def get_example(self, i) -> (numpy.ndarray, numpy.ndarray):
        image_path = self.paths[i]
        json_path = image_path.parent / (image_path.stem + ".json")
        json_data = json.load(json_path.open())
        face_meta = chainer_progressive_gan.models.face_extractor.FaceMeta.from_json(json_data)

        image = cv2.imread(str(image_path))
        original_image = image.copy()
        image = FaceBlendedDataset.to_blend_image(image1=image, image2=image.copy(), face_meta1=face_meta,
                                                  face_meta2=face_meta, gray=self.gray)
        #
        # margin = 0.25
        # image[face_meta.y:face_meta.y + face_meta.height, face_meta.x:face_meta.x + face_meta.width] = 0
        #
        # # subface
        # target_image = cv2.cvtColor(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY),
        #                             cv2.COLOR_GRAY2BGR) if self.gray else original_image
        # image[
        # face_meta.y + int(margin * face_meta.height):
        # face_meta.y + face_meta.height - int(margin * face_meta.height),
        # face_meta.x + int(margin * face_meta.width):
        # face_meta.x + face_meta.width - int(margin * face_meta.height)] = target_image[
        #                                                                   face_meta.y + int(
        #                                                                       margin * face_meta.height):
        #                                                                   face_meta.y + face_meta.height - int(
        #                                                                       margin * face_meta.height),
        #                                                                   face_meta.x + int(
        #                                                                       margin * face_meta.width):
        #                                                                   face_meta.x + face_meta.width - int(
        #                                                                       margin * face_meta.height)]
        if self.resize is None:
            return self._todata(image), self._todata(original_image)
        else:
            return self._todata(cv2.resize(image, self.resize)), self._todata(cv2.resize(original_image, self.resize))
