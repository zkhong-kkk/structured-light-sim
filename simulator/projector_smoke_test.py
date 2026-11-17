import blenderproc as bproc

import argparse
from pathlib import Path
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/projector_smoke")
args = parser.parse_args()

output_dir = Path(args.output).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

bproc.init()

bproc.renderer.set_render_devices(
    use_only_cpu=False,
    desired_gpu_device_type="OPTIX",
    desired_gpu_ids=[0],
)
bproc.renderer.set_max_amount_of_samples(32)
bproc.renderer.set_denoiser(None)
bproc.camera.set_resolution(640, 480)

# 白色漫反射物体
white = bproc.material.create("matte_white")
white.set_principled_shader_value("Base Color", [0.8, 0.8, 0.8, 1.0])
white.set_principled_shader_value("Roughness", 0.9)

cube = bproc.object.create_primitive("CUBE")
cube.set_scale([0.08, 0.08, 0.08])
cube.set_location([0.0, 0.0, 0.08])
cube.replace_materials(white)

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
periods = 12

x = np.arange(pattern_width, dtype=np.float32)
intensity = 0.5 + 0.5 * np.cos(
    2.0 * np.pi * periods * x / pattern_width
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

# 建立投影仪
projector = bproc.types.Light(light_type="SPOT", name="projector")
projector.set_energy(500.0)
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
