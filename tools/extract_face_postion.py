import argparse
import pathlib

import sys
import tqdm

thisfilepath = pathlib.Path(__file__)
sys.path.append(str(thisfilepath.parent.parent))
import chainer_progressive_gan


def main(args: argparse.Namespace):
    extractor = chainer_progressive_gan.models.FaceExtractor(
        cascade_file=str(thisfilepath.parent / "haarcascade_frontalface_default.xml"))
    for filepath in tqdm.tqdm(list(args.dataset_directory.iterdir())):
        try:
            target_path = args.output_directory / filepath.name
            if target_path.exists():
                continue
            extracted = extractor.extract_meta(img_file=str(filepath)).to_json()
            target_path.symlink_to(filepath)
            json_path = args.output_directory / (filepath.stem + ".json")
            json_path.open("w+").write(extracted)
        except Exception as e:
            if "face" in str(e):
                pass
            else:
                raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_directory', type=pathlib.Path)
    parser.add_argument('output_directory', type=pathlib.Path)
    args = parser.parse_args()
    main(args)
