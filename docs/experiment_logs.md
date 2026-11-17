# 仿真日志与证据索引

## 日志公开策略

`artifacts/logs/` 保存由服务器真实运行日志生成的脱敏副本。处理只替换用户名、用户目录和项目绝对路径，不改动数值、报错、版本、帧数或算法输出。服务器原始日志保留在本地工作目录 `logs/`，但不进入 Git。

| 公开日志 | 对应实验 | 可核验内容 |
| --- | --- | --- |
| `hole_camera2_pipeline.log` | 第二台同侧相机 12 视角流水线 | 144 帧渲染、12 视角重建、体积与拓扑结果 |
| `upper_lower_hole_pipeline.log` | 下路采集与上下双路融合 | 144 帧下路渲染、12 视角重建、联合 TSDF |
| `dual_tsdf_sweep.log` | 同侧双相机 TSDF 参数扫描 | 形态学核、连通域和拓扑失败结果 |
| `direct_poisson.log` | 上下双路直接点云 Poisson | 输入点数、降采样点数、网格规模 |
| `direct_poisson_tight.log` | 收紧 ROI 的 Poisson | 最终拓扑分支网格结果 |
| `tsdf_dual.log` | 双相机 TSDF | 每个相机/角度的积分过程 |
| `tsdf_fillfix.log` | TSDF 填充修正验证 | 小孔填充后的输出规模 |
| `upper_lower_roi.log` | 上下双路物理 ROI 重建 | ROI 后网格尺度恢复 |

## 指标证据

| 文件 | 内容 |
| --- | --- |
| `artifacts/results/random_cuboids_benchmark.csv` | 30 个随机长方体的尺寸、真值、原始/校正体积和误差 |
| `artifacts/results/random_cuboids_summary.json` | 由上述 CSV 重新计算的汇总统计 |
| `artifacts/results/hole_method_comparison.csv` | 单相机、同侧双相机、上下互补双路的统一对比 |
| `artifacts/results/upper_lower_metrics.txt` | 上下互补 TSDF 的原始服务器指标 |
| `artifacts/results/poisson_tight_metrics.txt` | 最终 Poisson 网格的原始服务器指标 |
| `artifacts/results/dual_tsdf_sweep.csv` | 同侧双相机参数扫描摘要 |

## 可审计原则

1. 报告中的汇总数字必须能由公开 CSV 重新计算；
2. 手工执行的截面拓扑检查必须给出命令或脚本入口；
3. 失败结果不从日志删除；
4. 路径脱敏不能改变时间、版本和指标；
5. 大体积原始数据不进入普通 Git，若后续需要发布应使用带校验和的数据集版本，而不是 Git LFS 隐式覆盖。

