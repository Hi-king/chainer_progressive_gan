# -*- coding: utf-8 -*-
import argparse
import glob
import os
import cv2
import PIL
import re
from PIL import Image
import json
import pipe
import math


def agif(cv_images, duration, filename):
    images = [Image.fromarray(cv_image[:, :, ::-1]) for cv_image in cv_images]
    images[0].save(filename, format="gif", save_all=True, duration=duration, append_images=images[1:], loop=0)


def main(args: argparse.Namespace):
    images = list(sorted(glob.glob(os.path.join(
        args.directory,
        'preview',
        'image[0-9]*.png'
    ))))

    log = json.load(open(os.path.join(
        args.directory,
        'log',
    )))

    cv_images = []
    for image_path in images:
        image = cv2.imread(image_path)
        iteration = int(re.match('.*image([0-9]+).png', image_path).group(1))

        meta = (log | pipe.where(lambda x: x['iteration'] >= iteration) | pipe.first)
        stage = meta['stage']

        w, h, _ = image.shape
        resolution = w // 10
        image = cv2.resize(image[:int(w // 10) * 3, :int(h // 10) * 3], (args.resize, args.resize),
                           interpolation=cv2.INTER_NEAREST)

        w, h, _ = image.shape
        cv2.putText(image, 'stage{}'.format(int(math.floor(stage)) + 1), (0, (h // 4) * 3), cv2.FONT_HERSHEY_PLAIN,
                    args.resize // 60,
                    (255, 128, 128), thickness=args.resize // 30)

        cv2.putText(image, '{}x{}'.format(resolution, resolution), (0, h), cv2.FONT_HERSHEY_PLAIN, args.resize // 60,
                    (255, 128, 255), thickness=args.resize // 30)
        cv_images.append(image)
    agif(cv_images, duration=args.duration, filename=os.path.join(
        args.directory,
        'preview.gif',
    ))


if __name__ == '__main__':
    assert (
        int(PIL.PILLOW_VERSION[0]) >= 4 or
        (int(PIL.PILLOW_VERSION[0]) == 3 and int(PIL.PILLOW_VERSION[2]) >= 4))  # agif feature
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--resize", type=int, default=256 * 3)
    parser.add_argument("--duration", type=int, default=1000, help="[ms]")
    args = parser.parse_args()
    main(args)
