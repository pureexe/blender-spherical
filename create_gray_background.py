# we use gray background instead of black background
# because we know that stable diffusion is not good
# when using too dark or too bright image

import argparse
import os 
import numpy as np 
import skimage
from multiprocessing import Pool 
from functools import partial
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='convert transparent background to gray background')

parser.add_argument("-s", "--source_dir", type=str, default="rgb_transparent")
parser.add_argument("-t", "--target_dir", type=str, default="rgb")

args = parser.parse_args()


def make_gray_bg(source_dir, target_dir, fname):
    """
    convert image file from transparent background to gray background
    """
    img = skimage.io.imread(os.path.join(source_dir, fname))
    img = skimage.img_as_float(img)
    # alpha blening with gray 
    for i in range(3):
        img[..., i] = (img[...,i] * img[..., 3]) + ((1.0 - img[...,3]) * 0.5)
    img = img[...,:3]
    img = skimage.img_as_ubyte(img)
    skimage.io.imsave(os.path.join(target_dir, fname), img)
    return None

def main():
    os.makedirs(args.target_dir, exist_ok=True)
    files = sorted(os.listdir(args.source_dir))
    fn = partial(make_gray_bg, args.source_dir, args.target_dir)
    with Pool(24) as p:
        _ = list(tqdm(p.imap(fn, files), total=len(files)))


if __name__ == "__main__":
    main()