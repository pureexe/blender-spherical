import os 
from tqdm.auto import tqdm
import argparse
import json
import math
import numpy as np
from dataset_info import DatasetInfo
# render image dataset
OUTPUT_DIR = "/data2/pakkapon/render/shoe401_envy_with_bg"
MOUNT_PREFIX = "/home/pakkapon/mnt_tl_vision22"
TOTAL_IMAGES = 60
USE_RANDOM_ENV_ANGLE = False
USE_RANDOM_CAM_ANGLE = False
GLB_PATH = MOUNT_PREFIX + "/data2/pakkapon/render/blender-spherical/assets/shoe401.glb" #set to none if want to randomly pick
HDRI_PATH = MOUNT_PREFIX + "/data2/pakkapon/render/blender-spherical/assets/abandoned_bakery_4k.exr" #set to none if want to randomly pick


parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument("-i", "--idx", type=int, required=True)
parser.add_argument("-t", "--total", type=int, required=True)

args = parser.parse_args()

def main():
    ds = DatasetInfo()
    output_dir = MOUNT_PREFIX+OUTPUT_DIR
    for image_id in tqdm(range(TOTAL_IMAGES)[args.idx::args.total]):
        
        if USE_RANDOM_ENV_ANGLE:
            env_vert, env_hori = ds.rand_env_angle()
        else:
            env_vert = (np.pi) / TOTAL_IMAGES  * image_id
            env_hori = 0.0

        if USE_RANDOM_CAM_ANGLE:
            cam_vert, cam_hori = ds.rand_cam_angle()
        else:
            cam_vert = np.pi / 2 
            cam_hori = 0.0
        
        if GLB_PATH is None:
            glb_path = ds.rand_glb()
        else:
            glb_path = GLB_PATH
        
        if HDRI_PATH is None:
            hdri_path = MOUNT_PREFIX + ds.rand_hdri()
        else:
            hdri_path = HDRI_PATH

        output_name = f'{image_id:06d}'

        validate_path = output_dir + f'/albedo/{output_name}.png'
        
        if not os.path.exists(validate_path):
            cmd = f"/home/vll/software/blender-3.2.2-linux-x64/blender --background --python blender_render.py -- --name {output_name} --object_path {glb_path} --envmap_path {hdri_path} --output_dir {output_dir} --env_vert {env_vert} --env_hori {env_hori} --cam_hori {cam_hori} --cam_vert {cam_vert} > /dev/null"                
            os.system(cmd)

if __name__ == "__main__":
    main()