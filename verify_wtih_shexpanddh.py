import pyshtools
import skimage
import numpy as np
from tqdm.auto import tqdm

COEFF_PATH = "assets/testing/axis_rendering_positive_cam/sh_rotated"

lmax = 100
output_image = []
idx = 0
loaded_coeff = np.load(f'{COEFF_PATH}/{idx:06d}.npy')
for ch in tqdm(range(3)):
    coeffs = loaded_coeff[ch]
    sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)
    coeff_grid = pyshtools.expand.MakeGridDH(sh_coeffs.coeffs, sampling=2, lmax=lmax)
    output_image.append(coeff_grid[...,None])

out_image = np.concatenate(output_image,axis=-1)
output_image = np.clip(output_image, 0.0 ,1.0)

output_image = skimage.img_as_ubyte(output_image)
skimage.io.imsave(f"verify_match.png", output_image)