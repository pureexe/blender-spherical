"""
Get information about hdris, objs
Note that: This one should only run on master node to reduce the comparasion
"""
# PRE-DEFINED CONFIG
PREFIX = "/home/pakkapon/mnt_tl_vision22"
POLYHAVEN_DIR = "/data/pakkapon/datasets/polyhaven"
OBJAVERSE_JSON_DIR = "/data/pakkapon/datasets/objaverse-relight-1m/assets/objaverse_path"

import math
import os
import json
import random

LIMIT_ENV_VERTICAL = [-math.pi / 2, math.pi / 2]
LIMIT_ENV_HORIZONTAL = [0, math.pi *2]
LIMIT_CAM_VERTICAL = [-math.pi / 2, math.pi / 2]
LIMIT_CAM_HORIZONTAL = [0, math.pi *2]

class DatasetInfo():
    def __init__(self):
        self.build_config()
        self.load_polyhaven_path()
        self.load_objaverse_path()

    def build_config(self):
        self.rotation_angles = [0, 90, 180, 270]
        self.envs_per_obj = 5
        self.num_envs = 500
        self.num_objs = 100000

    def count_images(self):
        return len(self.get_rotation_angle) * self.envs_per_obj * self.num_objs
    
    def load_polyhaven_path(self):
        polyjson = os.path.join(PREFIX + POLYHAVEN_DIR, "images", "scene.json")
        # verify that json file is exists
        if not os.path.exists(polyjson):
            raise Exception("PolyHaven directory is require images/scene.json make sure that the pointed directory is corrected ")
        
        with open(polyjson,'r') as f:
            scenes = []
            polyhaven_info = json.load(f)
            for p in polyhaven_info:
                scenes.append(p)
            scenes = sorted(scenes) 
            self.hdri_paths = [os.path.join(POLYHAVEN_DIR, 'images', 'exr', f"{s}_4k.exr") for s in scenes]


    def load_objaverse_path(self):
        files = sorted(os.listdir(PREFIX + OBJAVERSE_JSON_DIR))
        # filter only json file
        files = [f for f in files if f.endswith(".json")]

        if len(files) == 0:
            raise Exception("Objaverse_path should provide the json that contain key-value path that pointing to actual glb model")

        self.glb_paths = []
        self.glb_ids = []
        for json_file in files:
            with open(PREFIX + os.path.join(OBJAVERSE_JSON_DIR, json_file), 'r') as f:
                data = json.load(f)
                keys = list(data.keys())
                values = list(data.values())
                self.glb_paths = self.glb_paths + values
                self.glb_ids = self.glb_ids + keys

    def rand_hdri(self):
        #return random.choice(self.hdri_paths)
        return random.choice(self.hdri_paths[:self.num_envs])
    
    def rand_glb(self):
        #return random.choice(self.glb_paths)
        return random.choice(self.glb_paths[:self.num_objs])

    def rand_cam_angle(self):
        cam_vert =  (LIMIT_CAM_VERTICAL[0] + (LIMIT_CAM_VERTICAL[1] - LIMIT_CAM_VERTICAL[0]) * random.random())
        cam_hori =  (LIMIT_CAM_HORIZONTAL[0] + (LIMIT_CAM_HORIZONTAL[1] - LIMIT_CAM_HORIZONTAL[0]) * random.random())
        return cam_vert, cam_hori
    
    def rand_env_angle(self):
        env_vert =  (LIMIT_ENV_VERTICAL[0] + (LIMIT_ENV_HORIZONTAL[1] - LIMIT_ENV_HORIZONTAL[0]) * random.random())
        env_hori =  (LIMIT_ENV_HORIZONTAL[0] + (LIMIT_ENV_HORIZONTAL[1] - LIMIT_ENV_HORIZONTAL[0]) * random.random())
        return env_vert, env_hori
    
    """
    def get_rotation_angle(self, image_id):
        # should change to shifting rotation angle?
        # shift_angle = image_id // (len(self.rotation_angles)) // self.num_envs
        # return (self.rotation_angles[image_id % len(self.rotatiion_angle)] + shift_angle) % 360
        return self.rotation_angles[image_id % len(self.rotation_angles)]
    
    def get_glb_path(self, image_id):
        glb_id = image_id // (self.envs_per_obj * len(self.rotation_angles))
        if  glb_id > len(self.glb_paths) or glb_id > self.num_objs:
            raise Exception("You are request the object id more than avaible object. make sure that glb json and env_per_object are correct")
        return self.glb_paths[glb_id]
    
    def get_hdri_path(self, image_id):
        t_id = image_id // (len(self.rotation_angles))
        residual_obj =  t_id % self.envs_per_obj
        hdri_id = t_id // self.envs_per_obj
        hdri_id = hdri_id + residual_obj # shift by object_id
        hdri_id = hdri_id % self.num_envs # make it rotatable to avoid overflow
        if hdri_id > len(self.hdri_paths):
            raise Exception("You are request the hdri id more than avalible hdri. make sure that hdri json is correct")
        return self.hdri_paths[hdri_id]
    """