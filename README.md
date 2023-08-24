# blender-spherical
Render the image with spherical harmonic

The rotation code is under fine name rotate_the_sh_rotmat.py. 

that rotate from precompute SH value at 'assets/precompute_coeff/abandoned_bakery.npy' 

this rotation taken value from transform folder that got from the blender rendering code. 

blender rendering code is named 'blender_render.py' but it has to run with a parameter like envmap and rotation that will be send from 'render_distributor.py'


after done with rotated, we can check if everything correct by run 'visualize_sh.py' and the output image in folder 'sh_rotated_rotmat_view' should match the image that contain in folder 'rgb' 
