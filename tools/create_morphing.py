import argparse

import sys

import os

import chainer
import cv2
import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import models

def main(args):
    if args.resize == 32:
        channel_evolution = (512, 512, 512, 256)
    elif args.resize == 128:
        channel_evolution = (512, 512, 512, 512, 256, 128)
    elif args.resize == 256:
        channel_evolution = (512, 512, 512, 512, 256, 128, 64)  # too much memory
        # channel_evolution = (512, 512, 512, 256, 128, 64, 32)
    elif args.resize == 512:
        channel_evolution = (512, 512, 512, 512, 256, 128, 64, 32)
    elif args.resize == 1024:
        channel_evolution = (512, 512, 512, 512, 256, 128, 64, 32, 16)
    else:
        raise Exception()
    generator = models.progressive_generator.ProgressiveGenerator(channel_evolution=channel_evolution)
    chainer.serializers.load_npz(args.model_generator, generator)
    vectorizer = models.vectorizer.Vectorizer(channel_evolution=channel_evolution)
    chainer.serializers.load_npz(args.model_vectorizer, vectorizer)

    extractor = models.FaceExtractor(
        margin=args.margin
    )


    face_img = extractor.extract(args.target_img)
    # face_img = cv2.resize(cv2.cvtColor(cv2.imread(args.target_img), cv2.COLOR_BGR2RGB).astype(numpy.float32) / 256, (96, 96))
    face_img_var = chainer.Variable(
        numpy.array([face_img.transpose(2,0,1)*2 - 1.0]))

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        reconstructed = vectorizer(face_img_var)
        regenerated = generator(reconstructed)

    def clip_img(x):
        return numpy.float32(-1 if x<-1 else (1 if x>1 else x))

    def save(x, filepath):
        img = ((numpy.vectorize(clip_img)(x[0,:,:,:])+1)/2).transpose(1,2,0)
        print(img.shape)
        cv2.imwrite(
            filepath,
            cv2.cvtColor(img*256, cv2.COLOR_RGB2BGR)
        )

    os.makedirs(args.out_dir, exist_ok=True)
    save(face_img_var.data, os.path.join(args.out_dir, "face.png"))
    save(regenerated.data, os.path.join(args.out_dir, "reconstructed.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("target_img")
    parser.add_argument('--model_generator', type=str, required=True)
    parser.add_argument('--model_vectorizer', type=str, required=True)
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument("--out_dir", default="morphing")
    args = parser.parse_args()

    main(args)
