# 公开结果说明

这些文件是报告所引用的最小可审计结果集。除 `hole_method_comparison.csv` 和汇总 JSON 外，其余文本/CSV 均由服务器输出原样复制；汇总文件可由 `tools/prepare_public_artifacts.py` 重新生成。

## 结果来源

- `random_cuboids_benchmark.csv`：`outputs/random_cuboids/benchmark_results.csv`
- `random_cuboids_summary.json`：由上述 CSV 重新计算
- `upper_lower_metrics.txt`：上下互补 TSDF 服务器指标
- `poisson_tight_metrics.txt`：收紧 ROI 后 Poisson 服务器指标
- `camera2_*_metrics.txt`：第二台同侧相机独立结果
- `dual_tsdf_sweep.csv`：同侧双相机参数扫描汇总
- `hole_method_comparison.csv`：统一整理的通孔三方法对照表
- `four_camera_experiments.csv`：四相机在水平孔、双孔凹槽和深竖孔上的统一结果
- `four_camera_summary.json`：四相机实验的结构化汇总与失败边界
- `fourcam_*_metrics.txt`：四相机 TSDF/Poisson 原始服务器指标
- `fourcam_*_status.txt`：576 帧渲染和 48 视角重建完成状态

通孔对照表中的单相机结果来自同一服务器运行后的截面/体积检查：原始体积 975.1960329 cm³，校正体积 1018.1435077 cm³，校正误差 1.0063004%，闭合孔曲线为 0。上下互补方案的孔尺寸来自 `x=0` 中截面检查：40.69369 × 40.53752 mm。

## 解释限制

- `watertight=False` 不等于结果文件损坏，而表示网格存在开边界；
- TSDF 有向体积用于当前仿真比较，但实物阶段应增加网格闭合/体素交叉验证；
- Poisson 的 1179.58 cm³ 不是正式体积结论；
- `surface_offset_m` 是从标准立方体得到的仿真系统偏差，不可直接用于真实设备。
- 深竖孔场景中拓扑错误时，表面积校正会进一步放大体积误差；报告因此同时保留原始值和校正值，并将该场景判为失败。

