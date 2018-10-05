import argparse
import pathlib
import sys

import chainer
import cv2
import numpy

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parents[3]))
import chainer_progressive_gan
import tools


def main(args: argparse.Namespace):
    generator, _, _, vectorizer = tools.load_models(resize=args.resize, use_latent=args.use_latent,
                                                    pretrained_generator=args.generator,
                                                    pretrained_vectorizer=args.vectorizer)

    extractor = chainer_progressive_gan.models.FaceExtractor(
        cascade_file=str(thisfilepath.parent.parent.parent / "haarcascade_frontalface_default.xml"))
    image1 = cv2.imread(str(args.input_image1))
    image2 = cv2.imread(str(args.input_image2))
    face_meta_1 = extractor.extract_meta(img_file=str(args.input_image1))
    face_meta_2 = extractor.extract_meta(img_file=str(args.input_image2))

    # blended = chainer_progressive_gan.datasets.FaceBlendedDataset.to_blend_image(
    #     image1=image1.copy(), image2=image1.copy(), face_meta1=face_meta_1, face_meta2=face_meta_1)
    #
    # margin = 0.4
    # blended_face = blended[
    #                max(0, face_meta_1.y - int(margin * face_meta_1.height)):
    #                min(blended.shape[0], face_meta_1.y + face_meta_1.height + int(margin * face_meta_1.height)),
    #                max(0, face_meta_1.x - int(margin * face_meta_1.width)):
    #                min(blended.shape[1], face_meta_1.x + face_meta_1.width + int(margin * face_meta_1.height))]
    #
    # original_face = image1[
    #                max(0, face_meta_1.y - int(margin * face_meta_1.height)):
    #                min(blended.shape[0], face_meta_1.y + face_meta_1.height + int(margin * face_meta_1.height)),
    #                max(0, face_meta_1.x - int(margin * face_meta_1.width)):
    #                min(blended.shape[1], face_meta_1.x + face_meta_1.width + int(margin * face_meta_1.height))]
    #
    # cv2.imwrite("train.png", blended_face)
    # cv2.imwrite("original.png", original_face)
    # result_image = tools.utils.predict(blended_face, args.resize, args.stage, vectorizer, generator)
    # cv2.imwrite("result_face.png", result_image)

    blended = chainer_progressive_gan.datasets.FaceBlendedDataset.to_blend_image(
        image1=image1, image2=image2, face_meta1=face_meta_1, face_meta2=face_meta_2)

    margin = 0.4
    blended_face = blended[
                   max(0, face_meta_1.y - int(margin * face_meta_1.height)):
                   min(blended.shape[0], face_meta_1.y + face_meta_1.height + int(margin * face_meta_1.height)),
                   max(0, face_meta_1.x - int(margin * face_meta_1.width)):
                   min(blended.shape[1], face_meta_1.x + face_meta_1.width + int(margin * face_meta_1.height))]

    cv2.imwrite("input.png", blended_face)
    result_image = tools.utils.predict(blended_face, args.resize, args.stage, vectorizer, generator)
    result_image = cv2.resize(result_image, blended_face.shape[:2][::-1])
    cv2.imwrite("result_face.png", result_image)

    blended[
    max(0, face_meta_1.y - int(margin * face_meta_1.height)):
    min(blended.shape[0], face_meta_1.y + face_meta_1.height + int(margin * face_meta_1.height)),
    max(0, face_meta_1.x - int(margin * face_meta_1.width)):
    min(blended.shape[1], face_meta_1.x + face_meta_1.width + int(margin * face_meta_1.height))] = result_image

    cv2.imwrite("result.png", blended)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image1', type=pathlib.Path, required=True)
    parser.add_argument('--input_image2', type=pathlib.Path, required=True)

    parser.add_argument('--vectorizer', type=pathlib.Path, required=True)
    parser.add_argument('--generator', type=pathlib.Path, required=True)
    parser.add_argument('--use_latent', action=argparse._StoreTrueAction)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--stage', type=int, default=1000)
    args = parser.parse_args()
    main(args)
