import pyshtools as pysh
from scipy.spatial.transform import Rotation as R
import numpy as np
from tqdm.auto import tqdm
import os
import json

SH_IN_FOLDER = "assets/precompute_coeff/"
SH_OUT_FOLDER = "sh_rotated_rotmat"

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

def rotate_sh_rotmat(sh_coeff, rot, max_level = 2):
    # convert angle to zyz
    rot = R.from_matrix(rot)
    zyz = rot.as_euler('zyz', degrees=False)
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
    FOR_DEBUG = True

    nlat, nlon = 180, 360
    
    lmax = 100 if FOR_DEBUG else 2

    os.makedirs(SH_OUT_FOLDER, exist_ok=True)

    files = sorted(os.listdir("transform"))
    for fname in tqdm(files):
        # get env name
        with open(os.path.join("transform", fname)) as f:
            data = json.load(f)
            envname = data['env'].split("/")[-1][:-7]
            # convert to ROTVEC
            
            r_camoffset = R.from_euler('xyz', [0, np.pi / 2, np.pi]).as_matrix()

            r_cam = R.from_euler('xyz', [0, -data['cam_vert'], -data['cam_hori']]).as_matrix()
            r_env = R.from_euler('xyz', [0, -data['env_vert'], -data['env_hori']]).as_matrix()

            r_cam = r_camoffset @ r_cam 
            r_rot =  r_cam @ r_env

            main_sh = np.load(os.path.join(SH_IN_FOLDER, envname+".npy"))
            new_sh = rotate_sh_rotmat(main_sh.copy(), r_rot, lmax)
            
            sph_flatted = new_sh if FOR_DEBUG else flatten_sh_coeff(new_sh.copy(), 2)
            
            outname = fname.split(".")[0]
            np.save(os.path.join(SH_OUT_FOLDER, f"{outname}.npy"), sph_flatted)

   
if __name__ == "__main__":
    main()