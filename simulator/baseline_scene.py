import blenderproc as bproc

import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default="outputs/baseline",
    help="Output directory",
)
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

bproc.init()

# 优先使用服务器GPU
bproc.renderer.set_render_devices(
    use_only_cpu=False,
    desired_gpu_device_type=["OPTIX", "CUDA"],
    desired_gpu_ids=[0],
)
bproc.renderer.set_max_amount_of_samples(32)
# 相移仿真不能使用图像去噪
bproc.renderer.set_denoiser(None)

# 初始低分辨率，先验证流程
bproc.camera.set_resolution(640, 480)

# 创建白色亚光材料
matte_white = bproc.material.create("matte_white")
matte_white.set_principled_shader_value(
    "Base Color",
    [0.8, 0.8, 0.8, 1.0],
)
matte_white.set_principled_shader_value("Roughness", 0.85)

# 创建边长100 mm的立方体
cube = bproc.object.create_primitive("CUBE")
cube.set_scale([0.05, 0.05, 0.05])
cube.set_location([0.0, 0.0, 0.05])
cube.replace_materials(matte_white)

# 创建地面
ground = bproc.object.create_primitive("PLANE")
ground.set_scale([1.0, 1.0, 1.0])
ground.set_location([0.0, 0.0, 0.0])

ground_material = bproc.material.create("ground_material")
ground_material.set_principled_shader_value(
    "Base Color",
    [0.2, 0.2, 0.2, 1.0],
)
ground_material.set_principled_shader_value("Roughness", 0.9)
ground.replace_materials(ground_material)

# 临时面积光源；之后会替换成独立投影仪
light = bproc.types.Light()
light.set_type("AREA")
light.set_location([0.0, -0.5, 0.8])
light.set_energy(300.0)

# 设置相机
camera_location = np.array([0.0, -0.7, 0.35])
look_at = np.array([0.0, 0.0, 0.05])

rotation = bproc.camera.rotation_from_forward_vec(
    look_at - camera_location
)

camera_pose = bproc.math.build_transformation_mat(
    camera_location,
    rotation,
)

bproc.camera.add_camera_pose(camera_pose)

# 输出深度和法向真值
bproc.renderer.enable_depth_output(
    activate_antialiasing=False
)
bproc.renderer.enable_normals_output()

data = bproc.renderer.render()

bproc.writer.write_hdf5(
    str(output_dir),
    data,
)

print(f"Saved baseline output to: {output_dir.resolve()}")
