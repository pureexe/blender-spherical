import numpy as np
import matplotlib.pyplot as plt
import pyshtools
from tqdm.auto import tqdm 
import json
import skimage
import os

# Define the parameters for the image and the spherical harmonics
nside = 512  # Resolution of the spherical harmonic coefficients
lmax = 100   # Maximum degree of the spherical harmonics

# read fov
with open("fov.json") as f:
    fov = json.load(f)
    fov = fov['fov']
    hfov = fov / 2

flen = len(os.listdir("transform"))
flen = 200 if flen > 200 else flen

for idx in tqdm(range(flen)):
    # Generate random spherical harmonic coefficients
    loaded_coeff = np.load(f'sh_rotated_rotmat/{idx:06d}.npy')

    DIR = "sh_rotated_rotmat_view"
    os.makedirs(DIR, exist_ok=True)

    output_image = []
    for ch in tqdm(range(3)):
        coeffs = loaded_coeff[ch]

        
        
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

        # Create a grid of latitudes and longitudes
        #theta = np.linspace(np.pi/2, -np.pi /2, nside)
        #phi = np.linspace(0, np.pi * 2, 2*nside)
        theta = np.linspace(hfov, -hfov, nside)
        phi = np.linspace(np.pi - hfov, np.pi + hfov, nside)
        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)

    
        output_image.append(grid_data[...,None])

    output_image = np.concatenate(output_image,axis=-1)
    output_image = np.clip(output_image, 0.0 ,1.0)

    output_image = skimage.img_as_ubyte(output_image)
    skimage.io.imsave(f"{DIR}/{idx:06d}.png", output_image)
