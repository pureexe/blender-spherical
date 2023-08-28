# render test case for testing that everything is work according to the design

import numpy as np
import os
from tqdm.auto import tqdm 

OUTPUT_DIR = "/data2/pakkapon/render/blender-spherical/assets/testing/axis_rendering_positive_cam"
MOUNT_PREFIX = "/home/pakkapon/mnt_tl_vision22"
GLB_PATH = MOUNT_PREFIX + "/data2/pakkapon/render/blender-spherical/assets/shoe401.glb" #set to none if want to randomly pick
HDRI_PATH = MOUNT_PREFIX + "/data2/pakkapon/render/blender-spherical/assets/abandoned_bakery_4k.exr" #set to none if want to randomly pick


def main():
    output_dir = MOUNT_PREFIX+OUTPUT_DIR

    rotations = [] 
    NUM_STEP = 30
    rotate_steps = (np.arange(NUM_STEP) / NUM_STEP) * (np.pi * 2) # Render from 0 to 2pi * (NUM-1/NUM) because i don't last one to match with the first one
    # case 01: rotate across x-axis
    rotations = rotations + [[s, 0.0, 0.0] for s in rotate_steps]
    # case 02: rotate across y-axis
    rotations = rotations + [[0.0, s, 0.0] for s in rotate_steps]
    # case 03: rotate across z-axis
    rotations = rotations + [[0.0, 0.0, s] for s in rotate_steps]
    # case 04: rotate both x and y axis
    rotations = rotations + [[s, s, 0.0] for s in rotate_steps]
    # case 05: rotate both x and z axis 
    rotations = rotations + [[s, 0.0, s] for s in rotate_steps]
    # case 06: rotate y,z axis 
    rotations = rotations + [[0.0, s, s] for s in rotate_steps]    
    # case 07: rotate all x,y,z axis 
    rotations = rotations + [[s, s, s] for s in rotate_steps]

    for image_id, rot in enumerate(tqdm(rotations)):

        output_name = f'{image_id:06d}'
        validate_path = output_dir + f'/albedo/{output_name}.png'
        
        #if not os.path.exists(validate_path):
        if True:
            cmd = f"/home/vll/software/blender-3.2.2-linux-x64/blender --background --python blender_render.py -- --name {output_name} --object_path {GLB_PATH} --envmap_path {HDRI_PATH} --output_dir {output_dir} --env_rx {rot[0]} --env_ry {rot[1]} --env_rz {rot[2]} --obj_rx 0.0 --obj_ry 0.0 --obj_rz 0.0 > /dev/null" 
            os.system(cmd)
        exit()
       
if __name__ == "__main__":
    main()