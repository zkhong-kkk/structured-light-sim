import blenderproc as bproc

import argparse
from pathlib import Path

import bpy
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    "--turntable-angle",
    type=float,
    default=0.0,
)

parser.add_argument(
    "--periods",
    type=int,
    default=12,
)

parser.add_argument(
    "--output",
    default="outputs/single_phase_curved",
)

parser.add_argument(
    "--phase-degree",
    type=float,
    required=True,
)

parser.add_argument(
    "--camera-id",
    choices=[
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
    ],
    required=True,
)

parser.add_argument(
    "--cpu",
    action="store_true",
)

parser.add_argument(
    "--samples",
    type=int,
    default=128,
)

args = parser.parse_args()

output_dir = Path(args.output).resolve()
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# BlenderProc initialization
# ============================================================

bproc.init()

if args.cpu:
    bproc.renderer.set_render_devices(
        use_only_cpu=True,
    )
else:
    bproc.renderer.set_render_devices(
        use_only_cpu=False,
        desired_gpu_device_type="OPTIX",
        desired_gpu_ids=[0],
    )

bproc.renderer.set_max_amount_of_samples(args.samples)
bproc.renderer.set_denoiser(None)

bproc.camera.set_resolution(640, 480)

bproc.renderer.set_output_format(
    file_format="OPEN_EXR",
    color_depth=16,
)


# ============================================================
# Materials
# ============================================================

white = bproc.material.create("matte_white")

white.set_principled_shader_value(
    "Base Color",
    [0.8, 0.8, 0.8, 1.0],
)

white.set_principled_shader_value(
    "Roughness",
    0.9,
)


# ============================================================
# Curved complex object
#
# Raised stepped cylinder
# + two eccentric vertical through holes
# + four supporting feet
#
# The supporting feet expose the lower openings of the holes
# to the lower cameras.
# ============================================================

segment_count = 128

support_height = 0.012

lower_radius = 0.060
lower_height = 0.060

upper_radius = 0.045
upper_height = 0.040

hole_radius = 0.012
hole_offset_x = 0.018

foot_size_x = 0.018
foot_size_y = 0.018
foot_height = support_height

theta = np.deg2rad(args.turntable_angle)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)


def rotate_xy(x_value, y_value):
    rotated_x = (
        cos_theta * x_value
        - sin_theta * y_value
    )

    rotated_y = (
        sin_theta * x_value
        + cos_theta * y_value
    )

    return rotated_x, rotated_y


def apply_boolean_difference(target, cutter, modifier_name):
    bpy.ops.object.select_all(action="DESELECT")

    target.blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = target.blender_obj

    modifier = target.blender_obj.modifiers.new(
        name=modifier_name,
        type="BOOLEAN",
    )

    modifier.operation = "DIFFERENCE"
    modifier.solver = "EXACT"
    modifier.object = cutter.blender_obj

    bpy.ops.object.modifier_apply(
        modifier=modifier.name,
    )

    target.blender_obj.select_set(False)


# ============================================================
# Create lower and upper cylindrical stages
# Blender cylinder default radius is 1 and depth is 2.
# ============================================================

lower_body = bproc.object.create_primitive(
    "CYLINDER",
    vertices=segment_count,
)

lower_body.set_name("curved_lower_body")

lower_body.set_scale(
    [
        lower_radius,
        lower_radius,
        lower_height / 2.0,
    ]
)

lower_body.set_location(
    [
        0.0,
        0.0,
        support_height + lower_height / 2.0,
    ]
)

lower_body.persist_transformation_into_mesh(
    location=False,
    rotation=False,
    scale=True,
)


upper_body = bproc.object.create_primitive(
    "CYLINDER",
    vertices=segment_count,
)

upper_body.set_name("curved_upper_body")

upper_body.set_scale(
    [
        upper_radius,
        upper_radius,
        upper_height / 2.0,
    ]
)

upper_body.set_location(
    [
        0.0,
        0.0,
        support_height
        + lower_height
        + upper_height / 2.0,
    ]
)

upper_body.persist_transformation_into_mesh(
    location=False,
    rotation=False,
    scale=True,
)


# ============================================================
# Create the two vertical hole cutters
# ============================================================

hole_cutters = []

for hole_index, local_x in enumerate(
    [-hole_offset_x, hole_offset_x]
):
    cutter = bproc.object.create_primitive(
        "CYLINDER",
        vertices=segment_count,
    )

    cutter.set_name(
        f"vertical_hole_cutter_{hole_index}"
    )

    cutter.set_scale(
        [
            hole_radius,
            hole_radius,
            0.080,
        ]
    )

    cutter.set_location(
        [
            local_x,
            0.0,
            support_height + 0.050,
        ]
    )

    cutter.persist_transformation_into_mesh(
        location=False,
        rotation=False,
        scale=True,
    )

    hole_cutters.append(cutter)


# ============================================================
# Subtract both holes from both cylindrical stages
# ============================================================

for body_index, body in enumerate(
    [lower_body, upper_body]
):
    for hole_index, cutter in enumerate(
        hole_cutters
    ):
        apply_boolean_difference(
            body,
            cutter,
            (
                f"hole_{hole_index}"
                f"_body_{body_index}"
            ),
        )


# Delete cutter objects after applying Boolean modifiers
for cutter in hole_cutters:
    bpy.data.objects.remove(
        cutter.blender_obj,
        do_unlink=True,
    )


# Apply turntable rotation and material
for body in [lower_body, upper_body]:
    body.set_rotation_euler(
        [0.0, 0.0, theta]
    )

    body.replace_materials(white)

    body.set_shading_mode(
        "AUTO",
        angle_value=35.0,
    )


# ============================================================
# Four supporting feet
# ============================================================

foot_centers = [
    (-0.035, -0.025),
    (-0.035, 0.025),
    (0.035, -0.025),
    (0.035, 0.025),
]

for foot_index, (
    local_x,
    local_y,
) in enumerate(foot_centers):
    rotated_x, rotated_y = rotate_xy(
        local_x,
        local_y,
    )

    foot = bproc.object.create_primitive(
        "CUBE"
    )

    foot.set_name(
        f"support_foot_{foot_index}"
    )

    foot.set_scale(
        [
            foot_size_x / 2.0,
            foot_size_y / 2.0,
            foot_height / 2.0,
        ]
    )

    foot.set_location(
        [
            rotated_x,
            rotated_y,
            foot_height / 2.0,
        ]
    )

    foot.set_rotation_euler(
        [0.0, 0.0, theta]
    )

    foot.replace_materials(white)


# ============================================================
# Analytical ground-truth volume
# ============================================================

main_volume_m3 = np.pi * (
    (
        lower_radius ** 2
        - 2.0 * hole_radius ** 2
    )
    * lower_height
    +
    (
        upper_radius ** 2
        - 2.0 * hole_radius ** 2
    )
    * upper_height
)

feet_volume_m3 = (
    len(foot_centers)
    * foot_size_x
    * foot_size_y
    * foot_height
)

ground_truth_volume_m3 = (
    main_volume_m3
    + feet_volume_m3
)

ground_truth_volume_cm3 = (
    ground_truth_volume_m3
    * 1e6
)

print(
    "main_volume_cm3=",
    main_volume_m3 * 1e6,
)

print(
    "feet_volume_cm3=",
    feet_volume_m3 * 1e6,
)

print(
    "ground_truth_volume_cm3=",
    ground_truth_volume_cm3,
)

print(
    "expected_hole_diameter_mm=",
    hole_radius * 2.0 * 1000.0,
)


# ============================================================
# Ground plane
# ============================================================

ground_mat = bproc.material.create("ground")

ground_mat.set_principled_shader_value(
    "Base Color",
    [0.08, 0.08, 0.08, 1.0],
)

ground_mat.set_principled_shader_value(
    "Roughness",
    0.95,
)

ground = bproc.object.create_primitive("PLANE")
ground.set_scale([0.6, 0.6, 0.6])
ground.replace_materials(ground_mat)


# Keep the same target as the existing four-camera calibration.
target = np.array(
    [0.0, 0.0, 0.05]
)


# ============================================================
# Four-camera rig
# ============================================================

rigs = {
    "upper_left": {
        "camera": [-0.07, -0.07, 0.50],
        "projector": [0.0, 0.08, 0.52],
    },
    "upper_right": {
        "camera": [0.07, -0.07, 0.50],
        "projector": [0.0, 0.08, 0.52],
    },
    "lower_left": {
        "camera": [-0.28, -0.42, 0.025],
        "projector": [0.0, -0.48, 0.015],
    },
    "lower_right": {
        "camera": [0.28, -0.42, 0.025],
        "projector": [0.0, -0.48, 0.015],
    },
}

rig = rigs[args.camera_id]

camera_location = np.array(
    rig["camera"],
    dtype=np.float64,
)

camera_rotation = (
    bproc.camera.rotation_from_forward_vec(
        target - camera_location
    )
)

camera_pose = (
    bproc.math.build_transformation_mat(
        camera_location,
        camera_rotation,
    )
)

bproc.camera.add_camera_pose(camera_pose)


# ============================================================
# Sinusoidal structured-light pattern
# ============================================================

pattern_width = 640
pattern_height = 480

periods = args.periods
phase_rad = np.deg2rad(
    args.phase_degree
)

x = np.arange(
    pattern_width,
    dtype=np.float32,
)

intensity = (
    0.5
    + 0.5
    * np.cos(
        2.0
        * np.pi
        * periods
        * x
        / pattern_width
        + phase_rad
    )
)

gray = np.repeat(
    intensity[np.newaxis, :],
    pattern_height,
    axis=0,
)

pattern = np.stack(
    [
        gray,
        gray,
        gray,
        np.ones_like(gray),
    ],
    axis=-1,
).astype(np.float32)

print(
    "phase_degree=",
    args.phase_degree,
    "pattern_min=",
    pattern.min(),
    "pattern_max=",
    pattern.max(),
    "pattern_sum=",
    pattern.sum(),
    "first_values=",
    pattern[0, :5, 0],
)


# ============================================================
# Projector
# ============================================================

projector = bproc.types.Light(
    light_type="SPOT",
    name="projector",
)

projector.set_energy(200.0)
projector.setup_as_projector(pattern)

# Remove the automatic camera constraint.
for constraint in list(
    projector.blender_obj.constraints
):
    if constraint.type == "COPY_TRANSFORMS":
        projector.blender_obj.constraints.remove(
            constraint
        )

projector_location = np.array(
    rig["projector"],
    dtype=np.float64,
)

projector_rotation = (
    bproc.camera.rotation_from_forward_vec(
        target - projector_location
    )
)

projector.set_location(
    projector_location
)

projector.set_rotation_mat(
    projector_rotation
)

projector.blender_obj.data.cycles.cast_shadow = True


# ============================================================
# Render and save
# ============================================================

bproc.renderer.enable_depth_output(
    activate_antialiasing=False
)

data = bproc.renderer.render()

bproc.writer.write_hdf5(
    str(output_dir),
    data,
)

print(
    f"Saved curved object capture to: {output_dir}"
)
