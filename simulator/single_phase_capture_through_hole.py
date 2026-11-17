import blenderproc as bproc

import argparse
from pathlib import Path
import numpy as np
import bpy

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
    default="outputs/single_phase",
)

parser.add_argument(
    "--phase-degree",
    type=float,
    required=True,
)

args = parser.parse_args()

output_dir = Path(args.output).resolve()
output_dir.mkdir(parents=True, exist_ok=True)


bproc.init()

bproc.renderer.set_render_devices(
    use_only_cpu=False,
    desired_gpu_device_type="OPTIX",
    desired_gpu_ids=[0],
)
bproc.renderer.set_max_amount_of_samples(128)
bproc.renderer.set_denoiser(None)
bproc.camera.set_resolution(640, 480)
bproc.renderer.set_output_format(
    file_format="OPEN_EXR",
    color_depth=16,
)
# 白色漫反射物体
white = bproc.material.create("matte_white")
white.set_principled_shader_value("Base Color", [0.8, 0.8, 0.8, 1.0])
white.set_principled_shader_value("Roughness", 0.9)

# 方形通孔测试件
outer_x = 0.12
outer_y = 0.10
outer_z = 0.10

hole_x = 0.04
hole_y = 0.04

side_width = (outer_x - hole_x) / 2.0
front_width = (outer_y - hole_y) / 2.0

# 使用四个长方体围出贯穿方孔，避免布尔运算不稳定
parts = [
    # size_x, size_y, center_x, center_y
    (
        side_width,
        outer_y,
        -(hole_x + side_width) / 2.0,
        0.0,
    ),
    (
        side_width,
        outer_y,
        +(hole_x + side_width) / 2.0,
        0.0,
    ),
    (
        hole_x,
        front_width,
        0.0,
        -(hole_y + front_width) / 2.0,
    ),
    (
        hole_x,
        front_width,
        0.0,
        +(hole_y + front_width) / 2.0,
    ),
]

theta = np.deg2rad(args.turntable_angle)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

for index, (
    size_x,
    size_y,
    center_x,
    center_y,
) in enumerate(parts):
    block = bproc.object.create_primitive("CUBE")
    block.set_name(f"through_hole_part_{index}")

    block.set_scale(
        [
            size_x / 2.0,
            size_y / 2.0,
            outer_z / 2.0,
        ]
    )

    rotated_x = (
        cos_theta * center_x
        - sin_theta * center_y
    )
    rotated_y = (
        sin_theta * center_x
        + cos_theta * center_y
    )

    block.set_location(
        [
            rotated_x,
            rotated_y,
            outer_z / 2.0,
        ]
    )

    block.set_rotation_euler(
        [0.0, 0.0, theta]
    )

    block.replace_materials(white)
# 暗色背景平面
ground_mat = bproc.material.create("ground")
ground_mat.set_principled_shader_value("Base Color", [0.08, 0.08, 0.08, 1.0])
ground_mat.set_principled_shader_value("Roughness", 0.95)

ground = bproc.object.create_primitive("PLANE")
ground.set_scale([0.6, 0.6, 0.6])
ground.replace_materials(ground_mat)

target = np.array([0.0, 0.0, 0.08])

# 独立相机
camera_location = np.array([0.0, -0.65, 0.30])
camera_rotation = bproc.camera.rotation_from_forward_vec(
    target - camera_location
)
camera_pose = bproc.math.build_transformation_mat(
    camera_location, camera_rotation
)
bproc.camera.add_camera_pose(camera_pose)

# 生成竖直正弦条纹：12个周期
pattern_width = 640
pattern_height = 480
periods = args.periods
phase_rad = np.deg2rad(args.phase_degree)
x = np.arange(pattern_width, dtype=np.float32)
intensity = 0.5 + 0.5 * np.cos(
    2.0 * np.pi * periods * x / pattern_width
    + phase_rad
)

gray = np.repeat(
    intensity[np.newaxis, :],
    pattern_height,
    axis=0,
)

pattern = np.stack(
    [gray, gray, gray, np.ones_like(gray)],
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
# 建立投影仪
projector = bproc.types.Light(light_type="SPOT", name="projector")
projector.set_energy(200.0)
projector.setup_as_projector(pattern)

# setup_as_projector 默认把投影仪绑定到相机，删除该约束
for constraint in list(projector.blender_obj.constraints):
    if constraint.type == "COPY_TRANSFORMS":
        projector.blender_obj.constraints.remove(constraint)

# 设置独立投影仪位姿
projector_location = np.array([0.20, -0.45, 0.32])
projector_rotation = bproc.camera.rotation_from_forward_vec(
    target - projector_location
)

projector.set_location(projector_location)
projector.set_rotation_mat(projector_rotation)

# 结构光必须保留遮挡和阴影
projector.blender_obj.data.cycles.cast_shadow = True

bproc.renderer.enable_depth_output(
    activate_antialiasing=False
)

data = bproc.renderer.render()
bproc.writer.write_hdf5(str(output_dir), data)

print(f"Saved projector test to: {output_dir}")
