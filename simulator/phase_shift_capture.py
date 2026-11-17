import blenderproc as bproc

import argparse
from pathlib import Path
import bpy
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/phase_shift")
args = parser.parse_args()

output_dir = Path(args.output).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

bproc.init()

bproc.renderer.set_render_devices(
    use_only_cpu=False,
    desired_gpu_device_type="OPTIX",
    desired_gpu_ids=[0],
)
bproc.renderer.set_max_amount_of_samples(64)
bproc.renderer.set_denoiser(None)
bproc.camera.set_resolution(640, 480)

white = bproc.material.create("matte_white")
white.set_principled_shader_value(
    "Base Color", [0.8, 0.8, 0.8, 1.0]
)
white.set_principled_shader_value("Roughness", 0.9)

cube = bproc.object.create_primitive("CUBE")
cube.set_scale([0.08, 0.08, 0.08])
cube.set_location([0.0, 0.0, 0.08])
cube.replace_materials(white)

ground_material = bproc.material.create("ground_material")
ground_material.set_principled_shader_value(
    "Base Color", [0.08, 0.08, 0.08, 1.0]
)
ground_material.set_principled_shader_value("Roughness", 0.95)

ground = bproc.object.create_primitive("PLANE")
ground.set_scale([0.6, 0.6, 0.6])
ground.replace_materials(ground_material)

target = np.array([0.0, 0.0, 0.08])

camera_location = np.array([0.0, -0.65, 0.30])
camera_rotation = bproc.camera.rotation_from_forward_vec(
    target - camera_location
)
camera_pose = bproc.math.build_transformation_mat(
    camera_location, camera_rotation
)
bproc.camera.add_camera_pose(camera_pose)

pattern_width = 640
pattern_height = 480
periods = 12


def make_pattern(phase_rad):
    x = np.arange(pattern_width, dtype=np.float32)

    intensity = 0.5 + 0.5 * np.cos(
        2.0 * np.pi * periods * x / pattern_width + phase_rad
    )

    gray = np.repeat(
        intensity[np.newaxis, :],
        pattern_height,
        axis=0,
    )

    return np.stack(
        [gray, gray, gray, np.ones_like(gray)],
        axis=-1,
    ).astype(np.float32)


projector = bproc.types.Light(
    light_type="SPOT",
    name="projector",
)
projector.set_energy(500.0)
projector.setup_as_projector(make_pattern(0.0))

# 删除“投影仪跟随相机”的默认约束
for constraint in list(projector.blender_obj.constraints):
    if constraint.type == "COPY_TRANSFORMS":
        projector.blender_obj.constraints.remove(constraint)

projector_location = np.array([0.20, -0.45, 0.32])
projector_rotation = bproc.camera.rotation_from_forward_vec(
    target - projector_location
)

projector.set_location(projector_location)
projector.set_rotation_mat(projector_rotation)
projector.blender_obj.data.cycles.cast_shadow = True

pattern_image = bpy.data.images["pattern"]
pattern_image.colorspace_settings.name = "Non-Color"

bproc.renderer.enable_depth_output(
    activate_antialiasing=False
)

for phase_degree in [0, 90, 180, 270]:
    phase_rad = np.deg2rad(phase_degree)
    pattern = make_pattern(phase_rad)

    pattern_image.pixels.foreach_set(pattern.ravel())
    pattern_image.update()

    data = bproc.renderer.render()

    phase_dir = output_dir / f"phase_{phase_degree:03d}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    bproc.writer.write_hdf5(str(phase_dir), data)

    captured = data["colors"][0]
    Image.fromarray(captured).save(
        output_dir / f"capture_{phase_degree:03d}.png"
    )

    projected = (pattern[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(projected).save(
        output_dir / f"pattern_{phase_degree:03d}.png"
    )

    print(f"Finished phase {phase_degree} degrees")

print(f"Saved phase-shift sequence to: {output_dir}")
