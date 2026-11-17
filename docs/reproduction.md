# 复现实验指南

## 1. 环境

推荐 Ubuntu 22.04/24.04、Python 3.11 和 NVIDIA GPU。已验证版本：BlenderProc 2.8.0、Blender 4.2.1 LTS、Open3D 0.19.0。

```bash
cd /path/to/structured-light-sim

python3 -m venv /path/to/venvs/structured-light-sim
source /path/to/venvs/structured-light-sim/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果系统 `python3 -m venv` 缺少 `ensurepip`，安装与 Python 版本匹配的 `python3-venv`，或使用 uv/conda 创建 Python 3.11 环境。

## 2. 快速烟雾测试

```bash
CUDA_VISIBLE_DEVICES=0 blenderproc run \
  simulator/single_phase_capture.py \
  --periods 12 \
  --phase-degree 0 \
  --turntable-angle 0 \
  --output outputs/smoke
```

检查输出：

```bash
python -c "import h5py; f=h5py.File('outputs/smoke/0.hdf5'); print({k:(f[k].shape,str(f[k].dtype)) for k in f.keys()})"
```

## 3. 单个视角的三频四步采集

以水平通孔为例：

```bash
for frequency in 1 4 12; do
  for phase in 0 90 180 270; do
    CUDA_VISIBLE_DEVICES=0 blenderproc run \
      simulator/single_phase_capture_horizontal_hole.py \
      --periods "$frequency" \
      --phase-degree "$phase" \
      --turntable-angle 0 \
      --output "outputs/hole_view_000/freq_$(printf '%02d' "$frequency")/phase_$(printf '%03d' "$phase")"
  done
done
```

## 4. 多频解码与三角测量

```bash
python reconstruction/unwrap_multifrequency_16bit.py \
  --input outputs/hole_view_000 \
  --output outputs/hole_reconstruction/view_000/phase \
  --min-average 0.001 \
  --min-mod-ratio 0.05 \
  --saturation-level 1000000000 \
  --max-error 0.5 \
  --spatial-threshold 3.0 \
  --depth-cutoff 2.0

python reconstruction/triangulate.py \
  --phase outputs/hole_reconstruction/view_000/phase/unwrapped_phase_data.npz \
  --capture outputs/hole_view_000/freq_12/phase_000/0.hdf5 \
  --calibration configs/sim_calibration.npz \
  --projector-u-calibration configs/projector_u_calibration.npz \
  --output outputs/hole_reconstruction/view_000/reconstruction \
  --local-range-threshold 0.01 \
  --roi-radius 0.09 \
  --roi-min-z 0.003 \
  --roi-max-z 0.115
```

## 5. 12 视角上路和下路

批量脚本采用 12 个角度、3 个频率和 4 个相位，每一路生成 144 个 HDF5。

```bash
python simulator/batch_render_horizontal_hole.py \
  --gpu 0 \
  --views 0 30 60 90 120 150 180 210 240 270 300 330 \
  --output outputs/multiview_exr_horizontal_hole_12views

python simulator/batch_render_horizontal_hole_lower_pair.py \
  --gpu 0 \
  --views 0 30 60 90 120 150 180 210 240 270 300 330 \
  --output outputs/multiview_exr_horizontal_hole_lower_pair_12views
```

重建：

```bash
python reconstruction/batch_reconstruct_exr.py \
  --views 0 30 60 90 120 150 180 210 240 270 300 330 \
  --input outputs/multiview_exr_horizontal_hole_12views \
  --output outputs/multiview_reconstruction_exr_horizontal_hole_12views \
  --calibration configs/sim_calibration.npz

python reconstruction/batch_reconstruct_exr.py \
  --views 0 30 60 90 120 150 180 210 240 270 300 330 \
  --input outputs/multiview_exr_horizontal_hole_lower_pair_12views \
  --output outputs/multiview_reconstruction_exr_horizontal_hole_lower_pair_12views \
  --calibration configs/sim_calibration_lower_pair.npz
```

## 6. 上下互补 TSDF 体积分支

```bash
python reconstruction/tsdf_fusion_dual_camera.py \
  --views 0 30 60 90 120 150 180 210 240 270 300 330 \
  --camera1-reconstruction-root outputs/multiview_reconstruction_exr_horizontal_hole_12views \
  --camera1-capture-root outputs/multiview_exr_horizontal_hole_12views \
  --camera1-calibration configs/sim_calibration.npz \
  --camera2-reconstruction-root outputs/multiview_reconstruction_exr_horizontal_hole_lower_pair_12views \
  --camera2-capture-root outputs/multiview_exr_horizontal_hole_lower_pair_12views \
  --camera2-calibration configs/sim_calibration_lower_pair.npz \
  --output outputs/tsdf_horizontal_hole_upper_lower_12views_roi \
  --voxel-size 0.001 \
  --sdf-trunc 0.004 \
  --fill-kernel 5 \
  --fill-iterations 1 \
  --min-component-area 20 \
  --object-radius 0.09 \
  --min-z 0.003 \
  --max-z 0.115
```

`scripts/run_upper_lower_hole_pipeline.sh` 提供从下路渲染到双路 TSDF 的完整流水线。运行前请检查其中输入目录是否与本机实验一致。

流水线默认使用当前已激活环境；也可以显式指定环境目录：

```bash
STRUCTURED_LIGHT_VENV=/path/to/venvs/structured-light-sim \
  bash scripts/run_upper_lower_hole_pipeline.sh
```

## 7. 拓扑分支

```bash
python reconstruction/poisson_from_dual_points.py \
  --views 0 30 60 90 120 150 180 210 240 270 300 330 \
  --upper-root outputs/multiview_reconstruction_exr_horizontal_hole_12views \
  --upper-calibration configs/sim_calibration.npz \
  --lower-root outputs/multiview_reconstruction_exr_horizontal_hole_lower_pair_12views \
  --lower-calibration configs/sim_calibration_lower_pair.npz \
  --output outputs/poisson_horizontal_hole_direct_dual_tight \
  --object-radius 0.085 \
  --min-z 0.003 \
  --max-z 0.105 \
  --normal-radius 0.004 \
  --voxel-size 0.001 \
  --poisson-depth 10 \
  --poisson-scale 1.01
```

注意：Poisson 结果只检查孔截面与可视化，不输出正式体积。

统一评估：

```bash
python evaluation/evaluate_volume_mesh.py \
  --mesh outputs/tsdf_horizontal_hole_upper_lower_12views_roi/tsdf_mesh_raw.ply \
  --truth-cm3 1008 \
  --output outputs/tsdf_horizontal_hole_upper_lower_12views_roi/volume_evaluation.json

python evaluation/evaluate_hole_topology.py \
  --mesh outputs/poisson_horizontal_hole_direct_dual_tight/poisson_mesh_raw.ply \
  --plane-x 0 \
  --plane-z 0.05 \
  --expected-y 0.04 \
  --expected-z 0.04 \
  --output outputs/poisson_horizontal_hole_direct_dual_tight/topology_evaluation.json
```

## 8. 四相机 576 帧完整流水线

四相机脚本默认使用上左、上右、下左、下右四支路，12 个转台角度、3 个频率、4 个相位。运行前激活环境，并通过环境变量指定项目、环境和 GPU：

```bash
cd /path/to/structured-light-sim
source /path/to/venvs/structured-light-sim/bin/activate

export STRUCTURED_LIGHT_PROJECT="$PWD"
export STRUCTURED_LIGHT_VENV=/path/to/venvs/structured-light-sim
export STRUCTURED_LIGHT_GPU=0
```

水平单方孔：

```bash
nohup bash pipelines/run_fourcam_render.sh \
  > logs/fourcam_render_console.log 2>&1 &

# 渲染完成后
nohup bash pipelines/run_fourcam_reconstruction.sh \
  > logs/fourcam_reconstruction_console.log 2>&1 &
```

双方孔 + 顶部凹槽：

```bash
nohup bash pipelines/run_complex_fourcam_render.sh \
  > logs/complex_fourcam_render_console.log 2>&1 &

nohup bash pipelines/run_complex_fourcam_reconstruction.sh \
  > logs/complex_fourcam_reconstruction_console.log 2>&1 &
```

阶梯圆柱双深孔与陡俯视消融：

```bash
nohup bash pipelines/run_curved_fourcam_render.sh \
  > logs/curved_fourcam_render_console.log 2>&1 &

nohup bash pipelines/run_curved_fourcam_reconstruction.sh \
  > logs/curved_fourcam_reconstruction_console.log 2>&1 &

# 复用原下方两路，只重渲染 2×12×3×4=288 帧陡俯视上方两路
nohup bash pipelines/run_curved_steep_upper_render.sh \
  > logs/curved_steep_render_console.log 2>&1 &

nohup bash pipelines/run_curved_steep_reconstruction.sh \
  > logs/curved_steep_reconstruction_console.log 2>&1 &
```

流水线会写入 `outputs/*_status.txt`。只有状态为 `COMPLETE` 且对应重建视角数达到 48，才进入评价。断开 SSH 不影响 `nohup` 任务。

通用多孔截面评价示例：

```bash
python evaluation/evaluate_multi_hole_section.py \
  --mesh outputs/poisson_complex_fourcam_direct/poisson_mesh_raw.ply \
  --axis x \
  --position 0.0 \
  --expected-a 0.025 \
  --expected-b 0.035 \
  --min-extent 0.01 \
  --max-extent 0.06 \
  --output outputs/poisson_complex_fourcam_direct/topology_evaluation.json
```

若不使用封装流水线，`tsdf_fusion_multi_camera.py` 的每个 `--rig` 参数依次为标签、重建目录、采集目录、标定文件；`poisson_from_multi_points.py` 的每个 `--rig` 参数依次为标签、重建目录、标定文件。四个 `--rig` 可重复传入。

## 9. 随机长方体批量实验

```bash
python simulator/render_random_cuboids.py \
  --output-root outputs/random_cuboids \
  --count 30 \
  --seed 20261116

python reconstruction/run_cuboid_benchmark.py \
  --capture-root outputs/random_cuboids \
  --reconstruction-root outputs/random_cuboids_reconstruction \
  --tsdf-root outputs/random_cuboids_tsdf \
  --calibration configs/sim_calibration.npz \
  --volume-calibration configs/tsdf_volume_calibration.npz
```

实际参数名称以 `python <script> --help` 为准。批量结果应与 `artifacts/results/random_cuboids_benchmark.csv` 逐行比较，而不是只比较平均数。

## 10. 生成报告图表与公开证据

```bash
python tools/generate_report_assets.py
python tools/prepare_public_artifacts.py
```

生成的 SVG 直接用于 README 和仿真报告；第二个脚本从本地 `outputs/` 与 `logs/` 生成可进入 Git 的小型指标和脱敏日志，不复制 HDF5、PLY 等大文件。

## 11. 复现检查清单

- 四个相位的渲染图不能完全相同；
- 线性图像不得被错误截断到 `[0,1]` 后再量化；
- 三个频率目录必须是 `freq_01`、`freq_04`、`freq_12`；
- Open3D 使用 0.19.0；
- TSDF 输入深度单位是米，`depth_scale=1.0`；
- 转台角度进入融合前必须执行负角度回变换；
- 体积校正量仅适用于当前仿真配置；
- 非 watertight 网格和边界边数量必须记录；
- Poisson 体积不得作为本项目正式体积结论。
- 四相机完整对象必须有 576 个 HDF5 和 48 个重建视角；
- 目标孔数量、尺寸和中心位置必须独立于体积数字检查；
- 拓扑失败时不得继续信任表面积偏差校正后的体积。

