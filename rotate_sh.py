# rotate a spherical harmonic with json in transform files

import pyshtools as pysh
from scipy.spatial.transform import Rotation as R
import numpy as np
from tqdm.auto import tqdm
import os
import json

SH_IN_FOLDER = "assets/precompute_coeff/"
SH_OUT_FOLDER = "assets/testing/axis_rendering_positive_cam/sh_rotated"
TRANSFORM_DIR = "assets/testing/axis_rendering_positive_cam/transform"
FOR_DEBUG = True

def pysh_rotate(coeffs:np.array, angle:np.array, max_degree:int = 2):
    """
    Rotate spehrical harmonic
    Args:
        coeff: #shape[C,2, max_degree+1, max_degree+1]
        angle: 
    Returns:
        coeff: coeff
    """
    dj_mat = pysh.rotate.djpi2(max_degree)
    coeffs_rotated = []
    for c in range(coeffs.shape[0]):
        rotated = pysh.rotate.SHRotateRealCoef(coeffs[c],angle,dj_mat)
        coeffs_rotated.append(rotated[None])
    coeffs_rotated = np.concatenate(coeffs_rotated,axis=0)
    return coeffs_rotated

def rotate_sh(sh_coeff, rot, max_level = 2, invert_transform=False):
    """
    @params:
        sh_coeff: spherical harmonic in format [3,2,l+1,l+1]
        rot: Scipy's ortation class
        max_level: Max_level of spherical harmonic
    """
    # convert angle to zyz
    zyz = rot.as_euler('zyz', degrees=False)
    if invert_transform:
        # according to docs: the default behavior is a rotation of coordinatge system
        # but we can change to roataion of physical body by
        # The inverse transform of x(alpha, beta, gamma) is x(-gamma, -beta, -alpha).
        # @see https://shtools.github.io/SHTOOLS/pyshrotaterealcoef.html#description
        zyz[0], zyz[1], zyz[2] = -zyz[2], -zyz[1], -zyz[0]

    sh_coeff = pysh_rotate(sh_coeff, zyz, max_level)
    return sh_coeff

def flatten_sh_coeff(sh_coeff, max_sh_level = 2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    """
    flatted_coeff = np.zeros((3, (max_sh_level+1) ** 2))
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            for k in range(j+1):
                flatted_coeff[i, c] = sh_coeff[i, 0, j, k]
                c += 1
        for j in range(1, max_sh_level+1):
            for k in range(1, j+1):
                flatted_coeff[i, c] = sh_coeff[i, 1, j, k]
                c +=1
    return flatted_coeff


def main():
    
    lmax = 100 if FOR_DEBUG else 2

    os.makedirs(SH_OUT_FOLDER, exist_ok=True)

    files = sorted(os.listdir(TRANSFORM_DIR))
    for fname in tqdm(files):
        # get env name
        with open(os.path.join(TRANSFORM_DIR, fname)) as f:
            data = json.load(f)
            envname = data['env'].split("/")[-1][:-7]

            #r_rot = R.from_euler('xyz', [data['env_rx'], data['env_ry'], data['env_rz']])
            r_rot = R.from_euler('xyz', [data['env_rx'], data['env_ry'], data['env_rz']])
            
            #convert from blender to OPENGL convention
            # note: THIS IS NOT PERMUTATION we got this from blender's axis_conversion(to_forward="-Z", to_up="Y")
            # Some confirmation: https://stackoverflow.com/a/27887433
            #r_rot = R.from_euler('xyz', [data['env_rx'], data['env_rz'], -data['env_ry']])

            main_sh = np.load(os.path.join(SH_IN_FOLDER, envname+".npy"))
            new_sh = rotate_sh(main_sh.copy(), r_rot, lmax, invert_transform=True)
            
            sph_flatted = new_sh if FOR_DEBUG else flatten_sh_coeff(new_sh.copy(), 2)
            
            outname = fname.split(".")[0]
            np.save(os.path.join(SH_OUT_FOLDER, f"{outname}.npy"), sph_flatted)

   
if __name__ == "__main__":
    main()