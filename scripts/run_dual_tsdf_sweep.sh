#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
if [[ -n "${STRUCTURED_LIGHT_VENV:-}" ]]; then
  source "${STRUCTURED_LIGHT_VENV}/bin/activate"
fi

STATUS_FILE="outputs/dual_tsdf_sweep_status.txt"
SWEEP_ROOT="outputs/dual_tsdf_sweep"

mkdir -p "${SWEEP_ROOT}" logs

echo "RUNNING $(date --iso-8601=seconds)" > "${STATUS_FILE}"

on_error() {
  code=$?
  echo "FAILED $(date --iso-8601=seconds) code=${code}" \
    > "${STATUS_FILE}"
  exit "${code}"
}

trap on_error ERR

VIEWS=(
  0 30 60 90 120 150
  180 210 240 270 300 330
)

COMMON=(
  python reconstruction/tsdf_fusion_dual_camera.py
  --views "${VIEWS[@]}"
  --camera1-reconstruction-root
  outputs/multiview_reconstruction_exr_horizontal_hole_12views
  --camera1-capture-root
  outputs/multiview_exr_horizontal_hole_12views
  --camera1-calibration
  configs/sim_calibration.npz
  --camera2-reconstruction-root
  outputs/multiview_reconstruction_exr_horizontal_hole_camera2_12views
  --camera2-capture-root
  outputs/multiview_exr_horizontal_hole_camera2_12views
  --camera2-calibration
  configs/sim_calibration_camera2.npz
  --voxel-size 0.001
  --sdf-trunc 0.004
)

# 名称 min_component_area fill_kernel fill_iterations
CONFIGS=(
  "area1_kernel5_iter1 1 5 1"
  "area5_kernel5_iter1 5 5 1"
  "area1_kernel7_iter1 1 7 1"
  "area1_kernel9_iter1 1 9 1"
  "area1_kernel7_iter2 1 7 2"
)

for specification in "${CONFIGS[@]}"; do
  read -r name min_area kernel iterations \
    <<< "${specification}"

  output="${SWEEP_ROOT}/${name}"

  echo
  echo "============================================================"
  echo "Running ${name}"
  echo "min_component_area=${min_area}"
  echo "fill_kernel=${kernel}"
  echo "fill_iterations=${iterations}"
  echo "============================================================"

  "${COMMON[@]}" \
    --output "${output}" \
    --min-component-area "${min_area}" \
    --fill-kernel "${kernel}" \
    --fill-iterations "${iterations}"

  python -c "import trimesh,numpy as np; truth=1008.0; offset=float(np.load('configs/tsdf_volume_calibration.npz')['surface_offset_m']); full=trimesh.load_mesh('${output}/tsdf_mesh_raw.ply',process=False); components=list(full.split(only_watertight=False)); m=max(components,key=lambda x:len(x.faces)); trimesh.repair.fix_winding(m); trimesh.repair.fix_inversion(m,multibody=True); _,counts=np.unique(m.edges_sorted,axis=0,return_counts=True); raw=abs(m.volume)*1e6; corrected=(abs(m.volume)+offset*m.area)*1e6; section=m.section(plane_origin=[0.0,0.0,0.05],plane_normal=[1.0,0.0,0.0]); curves=[] if section is None else list(section.discrete); extents=[np.ptp(c[:,[1,2]],axis=0) for c in curves]; outer=any(np.all(np.abs(e-np.array([0.10,0.10]))<=np.array([0.015,0.015])) for e in extents); inner=any(np.all(np.abs(e-np.array([0.04,0.04]))<=np.array([0.012,0.012])) for e in extents); print('name=${name}'); print('components=',len(components)); print('main_ratio=',len(m.faces)/len(full.faces)); print('boundary_edges=',np.sum(counts==1)); print('raw_cm3=',raw); print('corrected_cm3=',corrected); print('corrected_error_percent=',abs(corrected-truth)/truth*100); print('section_entities=',0 if section is None else len(section.entities)); print('closed_curve_count=',len(curves)); print('curve_extents_yz_m=',extents); print('outer_loop_found=',outer); print('inner_loop_found=',inner); print('topology_pass=',outer and inner)" \
    | tee "${output}/sweep_metrics.txt"
done

echo "COMPLETE $(date --iso-8601=seconds)" > "${STATUS_FILE}"

echo
echo "Sweep completed."
