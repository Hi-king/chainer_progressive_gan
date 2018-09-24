import json
import os

import cv2
import numpy


class FaceMeta(object):
    def __init__(self, x, y, width, height, image_width, image_height):
        self.image_height = image_height
        self.image_width = image_width
        self.height = height
        self.width = width
        self.y = y
        self.x = x

    def to_json(self):
        return json.dumps({k: int(v) for k, v in self.__dict__.items()})

    @staticmethod
    def from_json(json_data):
        return FaceMeta(**json_data)


class FaceExtractor(object):
    def __init__(self, margin=0.3,
                 cascade_file=None):
        if cascade_file is None:
            cascade_file = os.path.join(os.path.dirname(__file__), "..", "animeface", "lbpcascade_animeface.xml")
        self.classifier = cv2.CascadeClassifier(cascade_file)
        self.margin = margin

    def extract_meta(self, img_file):
        target_img = cv2.imread(img_file)
        gray_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        gray_img_preprocessed = cv2.equalizeHist(gray_img)
        faces = self.classifier.detectMultiScale(
            gray_img_preprocessed,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24))
        if len(faces) == 0:
            raise (Exception("Could not find face from img"))
        image_width, image_height, _ = target_img.shape
        return FaceMeta(*faces[0], image_height=image_height, image_width=image_width)

    def extract(self, img_file):
        meta = self.extract_meta(img_file)
        target_img = cv2.imread(img_file)
        margin = int(min(
            meta.y, meta.image_height - meta.y - meta.width,
            meta.x, meta.image_width - meta.x - meta.width,
                    meta.width * self.margin
        ))
        rgb_img = cv2.cvtColor(cv2.resize(
            target_img[y - margin:meta.y + meta.height + margin, meta.x - margin:meta.x + meta.width + margin],
            (128, 128),
            # interpolation=cv2.INTER_LANCZOS4
            # interpolation=cv2.INTER_NEAREST,
            interpolation=cv2.INTER_AREA
        ), cv2.COLOR_BGR2RGB)

        float_img = rgb_img.astype(numpy.float32)
        return float_img / 256
