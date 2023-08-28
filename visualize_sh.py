import numpy as np
import matplotlib.pyplot as plt
import pyshtools
from tqdm.auto import tqdm 
import json
import skimage
import os

FOV_PATH = "assets/testing/axis_rendering_positive_cam/fov.json"
COEFF_PATH = "assets/testing/axis_rendering_positive_cam/sh_rotated"
TRANSFORM_PATH = "assets/testing/axis_rendering_positive_cam/transform"
OUT_DIR = "assets/testing/axis_rendering_positive_cam/sh_rotated_viz"
SHOW_ENTIRE_ENV_MAP = False

# Define the parameters for the image and the spherical harmonics
image_wide = 512  # Resolution of the spherical harmonic coefficients
lmax = 100   # Maximum degree of the spherical harmonics

# read fov
with open("fov.json") as f:
    fov = json.load(f)
    fov = fov['fov']
    hfov = fov / 2

flen = len(os.listdir(TRANSFORM_PATH))
flen = 300 if flen > 300 else flen

for idx in tqdm(range(61,flen)):
    # Generate random spherical harmonic coefficients
    loaded_coeff = np.load(f'{COEFF_PATH}/{idx:06d}.npy')

    os.makedirs(OUT_DIR, exist_ok=True)

    output_image = []
    for ch in tqdm(range(3)):
        coeffs = loaded_coeff[ch]
        
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

        # Create a grid of latitudes and longitudes
        if SHOW_ENTIRE_ENV_MAP:
            theta = np.linspace(np.pi / 2, -np.pi / 2, image_wide)
            phi = np.linspace(0, np.pi * 2, 2*image_wide)
        else:
            theta = np.linspace(hfov, -hfov, image_wide) #evaluation
            phi = np.linspace(-hfov, hfov, image_wide) #
        
        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)
        output_image.append(grid_data[...,None])

    output_image = np.concatenate(output_image,axis=-1)
    output_image = np.clip(output_image, 0.0 ,1.0)

    output_image = skimage.img_as_ubyte(output_image)
    skimage.io.imsave(f"{OUT_DIR}/{idx:06d}.png", output_image)
