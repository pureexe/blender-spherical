"""Blender script to render images of 3D models.

This script is used to render images of 3D models into an image together with the albedo
We also randomly select camera position and the rotation of environment map
output is in the form of a .png file with the following format:
    output_dir
        |- albedo
        |   |- [name].png
        |- rgb
        |   |- [name].png
        |- info
           |- [name].json

Example usage:
    blender -b -P blender_script.py -- \
        --name output_name \
        --object_path my_object.glb \
        --envmap_path my_envmap.hdr \
        --output_dir ./views \

"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import json

import bpy
from mathutils import Vector
from argparse import Namespace

IS_SCRIPTING_TAB = False #Enable this if use scripting tab to disable argparse

LIMIT_ENV_VERTICAL = [-math.pi / 2, math.pi / 2]
LIMIT_ENV_HORIZONTAL = [0, math.pi *2]
LIMIT_CAM_VERTICAL = [-math.pi / 2, math.pi / 2]
LIMIT_CAM_HORIZONTAL = [0, math.pi *2]
LIMIT_OBJ_VERTICAL = [-math.pi / 2, math.pi / 2]
LIMIT_OBJ_HORIZONTAL = [0, math.pi *2]

if not IS_SCRIPTING_TAB:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--envmap_path",
        type=str,
        required=True,
        help="Path to the envmap file",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Output name",
    )


    parser.add_argument("--output_dir", type=str, default="./views")
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--camera_dist", type=int, default=1.0) #previously we use 1.2
    parser.add_argument("--albedo_key", type=str, default="0001")
    parser.add_argument("--env_rx", type=float, default=0.0)
    parser.add_argument("--env_ry", type=float, default=0.0)
    parser.add_argument("--env_rz", type=float, default=0.0)
    parser.add_argument("--obj_rx", type=float, default=0.0)
    parser.add_argument("--obj_ry", type=float, default=0.0)
    parser.add_argument("--obj_rz", type=float, default=0.0)
    parser.add_argument('--has_background', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--negative_cam', action=argparse.BooleanOptionalAction, default=False)
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
else:
    args = {
        "object_path": "C:\\Users\\pakkapon\\Desktop\\blender\\shoe401.glb",
        "envmap_path": "C:\\Users\\pakkapon\\Desktop\\blender\\abandoned_bakery_4k.exr",
        "name": "test_output",
        "output_dir": "C:\\Users\\pakkapon\\Desktop\\blender\\output",
        "engine": "CYCLES",
        "num_images": 1,
        "camera_dist": 1.0,
        "env_rx": 0.0,
        "env_ry": 0.0,
        "env_rz": 0.0,
        "obj_rx": 0.0,
        "obj_ry": 0.0,
        "obj_rz": math.pi / 4,
        "albedo_key": "0001",
        "has_background": False,
        "negative_cam": True
    }
    args = Namespace(**args)

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = not args.has_background

def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    cuda_devices = cycles_preferences.devices[0]
    opencl_devices = cycles_preferences.devices[1]


    if device_type == "CUDA":
        devices = cuda_devices
    elif device_type == "OPENCL":
        devices = opencl_devices
    else:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []

    for device in [devices]:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus

enable_gpus("CUDA")

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent and obj.type not in {"CAMERA"}:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    """
    Normalize scene into a unit sphere
    """
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera(use_negative_cam=True):
    """ 
    place a camera at [-1,0,0] and looking to 
    note: previously we use cam_dist = 1.2 but change to 1.0 for easier computation
    """
    cam = scene.objects["Camera"]
    if use_negative_cam:
        cam.location = (-args.camera_dist, 0, 0) 
        # since camera is negative we need to flip the camera rotation from (90d, 0, 90d) to (90d, 0, -90d) (ROTATE ACROSS Z-up axis)
        cam.rotation_mode = 'XYZ'
        cam.rotation_euler.x = math.pi / 2
        cam.rotation_euler.y = 0
        cam.rotation_euler.z = -math.pi / 2
    else:
        cam.location = (args.camera_dist, 0, 0) 
        cam.rotation_mode = 'XYZ'
        cam.rotation_euler.x = math.pi / 2
        cam.rotation_euler.y = 0
        cam.rotation_euler.z = math.pi / 2
    
    cam.data.lens_unit = 'FOV' 
    cam.data.angle = math.atan2(0.5, args.camera_dist) * 2
    return cam 

def rotate_object(rot_x, rot_y, rot_z):
    for obj in scene_root_objects():
        #need to force set convetion to XYZ, spot some glb that use other convention
        obj.rotation_mode = 'XYZ' 
        obj.rotation_euler.x += rot_x
        obj.rotation_euler.y += rot_y
        obj.rotation_euler.z += rot_z

def add_environment(envmap_path, rot_x, rot_y, rot_z):
    """
    Add an environment map to shader node
    For shader node map, see https://i.imgur.com/sBlUXrB.png
    
    @params
        - ennvmap_path: Path of environment map (exr file)
        - rot_x: Rotation in X (Radiant)
        - rot_y: Rotation in Y (Radiant)
        - rot_z: Rotation in Z (Radiant)

    NOTE:
        Blender Roataion Convention (Euler XYZ, camera at z+ look to z-)
        Rx+: Clockwise
        Ry+: looking up
        Rz+: turn left
        However, since our camera at (-1, 0, 0)
        Rx+: Anti-Clockwise
        Ry+: looking down
        Rz+: turn left
    """
    nodes = bpy.data.worlds[0].node_tree.nodes
    links = bpy.data.worlds[0].node_tree.links
    node_color = nodes[1]

    node_env = nodes.new('ShaderNodeTexEnvironment')
    node_env.image = bpy.data.images.load(envmap_path)

    node_mapping = nodes.new('ShaderNodeMapping')
    node_mapping.inputs['Rotation'].default_value.x = rot_x
    node_mapping.inputs['Rotation'].default_value.y = rot_y # vertical
    node_mapping.inputs['Rotation'].default_value.z = rot_z #horizontal 

    node_coord = nodes.new('ShaderNodeTexCoord')
    links.new(node_mapping.outputs["Vector"], node_env.inputs["Vector"])
    links.new(node_env.outputs["Color"], node_color.inputs["Color"])
    links.new(node_coord.outputs["Generated"], node_mapping.inputs["Vector"])
    
def random_params():
    env_vert =  (LIMIT_ENV_VERTICAL[0] + (LIMIT_ENV_HORIZONTAL[1] - LIMIT_ENV_HORIZONTAL[0]) * random.random())
    env_hori =  (LIMIT_ENV_HORIZONTAL[0] + (LIMIT_ENV_HORIZONTAL[1] - LIMIT_ENV_HORIZONTAL[0]) * random.random())
    cam_vert =  (LIMIT_CAM_VERTICAL[0] + (LIMIT_CAM_VERTICAL[1] - LIMIT_CAM_VERTICAL[0]) * random.random())
    cam_hori =  (LIMIT_CAM_HORIZONTAL[0] + (LIMIT_CAM_HORIZONTAL[1] - LIMIT_CAM_HORIZONTAL[0]) * random.random())
    return env_vert, env_hori, cam_vert, cam_hori

def save_image(rgb_dir, albedo_dir, name):
    # render image path
    scene.render.filepath = os.path.join(rgb_dir, f"{name}.png")
    # albedo image path
    albedo_node = scene.node_tree.nodes['Albedo Output']
    albedo_node.base_path = albedo_dir
    albedo_node.file_slots[0].path = f"{name}_"
    bpy.ops.render.render(write_still=True)
    os.rename(os.path.join(albedo_dir, f"{name}_{args.albedo_key}.png"), os.path.join(albedo_dir, f"{name}.png"))

def add_albedo_rendering():
    """
    Add albedo composite node for rendering the albedo
    """
    bpy.context.scene.use_nodes = True
    bpy.context.view_layer.use_pass_diffuse_color = True
    tree = bpy.context.scene.node_tree
    render_node = None
    if 'Render Layers' in tree.nodes:
        render_node = tree.nodes['Render Layers']
    elif not 'Custom Ouputs' in tree.nodes:
        render_node = tree.nodes.new('CompositorNodeRLayers')
        render_node.label = 'Custom Ouputs'
        render_node.name = 'Custom Ouputs'
    
    if not 'Albedo Output' in tree.nodes:
        albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        albedo_file_output.label = 'Albedo Output'
        albedo_file_output.name = 'Albedo Output'
        albedo_file_output.format.file_format = 'PNG'
        tree.links.new(render_node.outputs['DiffCol'], albedo_file_output.inputs['Image'])

def main():
    rgb_dir = os.path.join(args.output_dir, "rgb")
    albedo_dir = os.path.join(args.output_dir, "albedo")
    json_dir = os.path.join(args.output_dir, "transform")
    os.makedirs(rgb_dir,exist_ok=True)
    os.makedirs(albedo_dir,exist_ok=True)
    os.makedirs(json_dir,exist_ok=True)

    reset_scene()
    load_object(args.object_path)
    normalize_scene()
    rotate_object(args.obj_rx, args.obj_ry, args.obj_rz)
    setup_camera(args.negative_cam)
    add_environment(args.envmap_path, args.env_rx, args.env_ry, args.env_rz)
    add_albedo_rendering()
    save_image(rgb_dir, albedo_dir, args.name)
    with open(os.path.join(json_dir, f"{args.name}.json"), "w") as f:
        data = {
            "glb": args.object_path,
            "env": args.envmap_path,
            "env_rx": args.env_rx,
            "env_ry": args.env_ry,
            "env_rz": args.env_rz,
            "obj_rx": args.obj_rx,
            "obj_ry": args.obj_ry,
            "obj_rz": args.obj_rz
        }
        json.dump(data,f,indent = 4)

if __name__ == "__main__" or IS_SCRIPTING_TAB:
    main()

