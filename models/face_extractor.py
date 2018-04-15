import cv2
import os

import numpy


class FaceExtractor(object):
    def __init__(self, margin=0.3,
                 cascade_file=None):
        if cascade_file is None:
            cascade_file = os.path.join(os.path.dirname(__file__), "..", "animeface", "lbpcascade_animeface.xml")
        self.classifier = cv2.CascadeClassifier(cascade_file)
        self.margin = margin

    def extract(self, img_file):
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

        x, y, width, height = faces[0]
        image_width, image_height, _ = target_img.shape
        margin = int(min(
            y, image_height - y - width,
            x, image_width - x - width,
               width * self.margin
        ))
        print(y - margin,y + height + margin, x - margin, x + width + margin)
        rgb_img = cv2.cvtColor(cv2.resize(
            target_img[y - margin:y + height + margin, x - margin:x + width + margin],
            (128, 128),
            # interpolation=cv2.INTER_LANCZOS4
            # interpolation=cv2.INTER_NEAREST,
            interpolation=cv2.INTER_AREA
        ), cv2.COLOR_BGR2RGB)

        float_img = rgb_img.astype(numpy.float32)
        return float_img / 256
