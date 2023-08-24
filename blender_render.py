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


LIMIT_ENV_VERTICAL = [-math.pi / 2, math.pi / 2]
LIMIT_ENV_HORIZONTAL = [0, math.pi *2]
LIMIT_CAM_VERTICAL = [-math.pi / 2, math.pi / 2]
LIMIT_CAM_HORIZONTAL = [0, math.pi *2]

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
parser.add_argument("--camera_dist", type=int, default=1.2)
parser.add_argument("--albedo_key", type=str, default="0001")
parser.add_argument("--env_vert", type=float, default=0.0)
parser.add_argument("--env_hori", type=float, default=0.0)
parser.add_argument("--cam_vert", type=float, default=math.pi / 2)
parser.add_argument("--cam_hori", type=float, default=0.0)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

"""
args = {
    "object_path": "C:\\Users\\pakkapon\\Desktop\\generate_objaverse\\01.glb",
    "envmap_path": "C:\\Users\\pakkapon\\Desktop\\generate_objaverse\\env.exr",
    "name": "test_output",
    "output_dir": "C:\\Users\\pakkapon\\Desktop\\generate_objaverse\\output",
    "engine": "CYCLES",
    "num_images": 1,
    "camera_dist": 10,
    "env_vert": 0,
    "env_hori": 0,
    "cam_vert": math.pi / 2,
    "cam_hori": math.pi / 2,
    "albedo_key": "0001"
}
args = Namespace(**args)
"""

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
scene.render.film_transparent = False

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
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
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


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, args.camera_dist, 0)
    #cam.data.lens = 35
    #cam.data.sensor_width = 32
    cam.data.lens_unit = 'FOV' 
    cam.data.angle = math.atan2(0.5, args.camera_dist) * 2
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint



def add_environment(vert_deg, hori_deg):
    nodes = bpy.data.worlds[0].node_tree.nodes
    links = bpy.data.worlds[0].node_tree.links
    node_color = nodes[1]

    node_env = nodes.new('ShaderNodeTexEnvironment')
    node_env.image = bpy.data.images.load(args.envmap_path)

    node_mapping = nodes.new('ShaderNodeMapping')
    node_mapping.inputs['Rotation'].default_value.y = vert_deg # vertical
    node_mapping.inputs['Rotation'].default_value.z = hori_deg #horizontal 

    node_coord = nodes.new('ShaderNodeTexCoord')
    links.new(node_mapping.outputs["Vector"], node_env.inputs["Vector"])
    links.new(node_env.outputs["Color"], node_color.inputs["Color"])
    links.new(node_coord.outputs["Generated"], node_mapping.inputs["Vector"])

def prepare_albedo():
    for material in bpy.data.materials:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        output_node = nodes['Material Output']
        bsdf_node = output_node.inputs['Surface'].links[0].from_node
        color_node = None
        bsdf_target_key = 'Base Color' if 'Base Color' in bsdf_node.inputs else 'Color'
        if len(bsdf_node.inputs[bsdf_target_key].links) > 0:
            bsdf_prev = bsdf_node.inputs[bsdf_target_key].links[0]
            color_node = bsdf_prev.from_node
            color_socket = bsdf_prev.from_socket.name
        else:
            # in case no node, we create new node
            color_node = nodes.new('ShaderNodeRGB')
            color_node.outputs['Color'].default_value = bsdf_node.inputs[bsdf_target_key].default_value
            color_socket = "Color"

        emit_node = nodes.new('ShaderNodeEmission')
        links.new(color_node.outputs[color_socket], emit_node.inputs["Color"])
        links.new(emit_node.outputs["Emission"], output_node.inputs["Surface"])

    bpy.context.scene.view_settings.view_transform = 'Standard'

    
def random_params():
    env_vert =  (LIMIT_ENV_VERTICAL[0] + (LIMIT_ENV_HORIZONTAL[1] - LIMIT_ENV_HORIZONTAL[0]) * random.random())
    env_hori =  (LIMIT_ENV_HORIZONTAL[0] + (LIMIT_ENV_HORIZONTAL[1] - LIMIT_ENV_HORIZONTAL[0]) * random.random())
    cam_vert =  (LIMIT_CAM_VERTICAL[0] + (LIMIT_CAM_VERTICAL[1] - LIMIT_CAM_VERTICAL[0]) * random.random())
    cam_hori =  (LIMIT_CAM_HORIZONTAL[0] + (LIMIT_CAM_HORIZONTAL[1] - LIMIT_CAM_HORIZONTAL[0]) * random.random())
    #env_vert, env_hori, cam_vert, cam_hori = 0.0, 0.0, math.pi/2, 0.0
    return env_vert, env_hori, cam_vert, cam_hori

def set_camera(cam_vert, cam_hori):
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    empty.location.x = 0
    empty.location.y = 0
    empty.location.z = 0
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    theta = cam_hori
    phi = cam_vert
    point = (
        args.camera_dist * math.sin(phi) * math.cos(theta),
        args.camera_dist * math.sin(phi) * math.sin(theta),
        args.camera_dist * math.cos(phi),
    )
    cam.location = point

def save_image(rgb_dir, albedo_dir, name):
    # render image path
    scene.render.filepath = os.path.join(rgb_dir, f"{name}.png")
    # albedo image path
    albedo_node = scene.node_tree.nodes['Albedo Output']
    albedo_node.base_path = albedo_dir
    albedo_node.file_slots[0].path = f"{name}_"
    bpy.ops.render.render(write_still=True)
    os.rename(os.path.join(albedo_dir, f"{name}_{args.albedo_key}.png"), os.path.join(albedo_dir, f"{name}.png"))

def add_custom_composite():
    # add custom composite node for rendering the albedo
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

    #env_vert, env_hori, cam_vert, cam_hori = random_params()
    env_vert, env_hori, cam_vert, cam_hori = args.env_vert, args.env_hori, args.cam_vert, args.cam_hori
    reset_scene()
    load_object(args.object_path)
    normalize_scene()
    set_camera(cam_vert, cam_hori)
    add_environment(env_vert, env_hori)
    add_custom_composite()
    save_image(rgb_dir, albedo_dir, args.name)
    with open(os.path.join(json_dir, f"{args.name}.json"), "w") as f:
        data = {
            "glb": args.object_path,
            "env": args.envmap_path,
            "env_vert": env_vert,
            "env_hori": env_hori,
            "cam_vert": cam_vert,
            "cam_hori": cam_hori
        }
        json.dump(data,f,indent = 4)

if __name__ == "__main__":
    main()

#main()