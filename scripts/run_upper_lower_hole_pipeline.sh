#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CAPTURE_ROOT="outputs/multiview_exr_horizontal_hole_lower_pair_12views"
RECON_ROOT="outputs/multiview_reconstruction_exr_horizontal_hole_lower_pair_12views"
TSDF_ROOT="outputs/tsdf_horizontal_hole_upper_lower_12views"

STATUS_FILE="outputs/upper_lower_hole_pipeline_status.txt"

VIEWS=(
  0 30 60 90 120 150
  180 210 240 270 300 330
)

cd "${PROJECT_ROOT}"
if [[ -n "${STRUCTURED_LIGHT_VENV:-}" ]]; then
  source "${STRUCTURED_LIGHT_VENV}/bin/activate"
fi
export CUDA_VISIBLE_DEVICES=0
mkdir -p \
  "${CAPTURE_ROOT}" \
  "${RECON_ROOT}" \
  "${TSDF_ROOT}" \
  logs
echo "RUNNING $(date --iso-8601=seconds)" > "${STATUS_FILE}"

on_error() {
  code=$?
  echo "FAILED $(date --iso-8601=seconds) code=${code}" \
    > "${STATUS_FILE}"
  exit "${code}"
}

trap on_error ERR

echo "============================================================"
echo "Stage 1: render lower pair"
echo "============================================================"

capture_count=$(
  find "${CAPTURE_ROOT}" -name "0.hdf5" 2>/dev/null | wc -l
)

if [ "${capture_count}" -ne 144 ]; then
  python simulator/batch_render_horizontal_hole_lower_pair.py \
    --gpu 0 \
    --views "${VIEWS[@]}" \
    --output "${CAPTURE_ROOT}"
fi

capture_count=$(
  find "${CAPTURE_ROOT}" -name "0.hdf5" | wc -l
)

echo "capture_count=${capture_count}"

if [ "${capture_count}" -ne 144 ]; then
  exit 20
fi

echo "============================================================"
echo "Stage 2: reconstruct lower pair"
echo "============================================================"

reconstruction_count=$(
  find "${RECON_ROOT}" \
    -name reconstruction_data.npz 2>/dev/null | wc -l
)

if [ "${reconstruction_count}" -ne 12 ]; then
  python reconstruction/batch_reconstruct_exr.py \
    --views "${VIEWS[@]}" \
    --input "${CAPTURE_ROOT}" \
    --output "${RECON_ROOT}" \
    --calibration configs/sim_calibration_lower_pair.npz
fi

reconstruction_count=$(
  find "${RECON_ROOT}" \
    -name reconstruction_data.npz | wc -l
)

echo "reconstruction_count=${reconstruction_count}"

if [ "${reconstruction_count}" -ne 12 ]; then
  exit 30
fi

echo "============================================================"
echo "Stage 3: upper and lower joint TSDF"
echo "============================================================"

python reconstruction/tsdf_fusion_dual_camera.py \
  --views "${VIEWS[@]}" \
  --camera1-reconstruction-root \
    outputs/multiview_reconstruction_exr_horizontal_hole_12views \
  --camera1-capture-root \
    outputs/multiview_exr_horizontal_hole_12views \
  --camera1-calibration \
    configs/sim_calibration.npz \
  --camera2-reconstruction-root \
    "${RECON_ROOT}" \
  --camera2-capture-root \
    "${CAPTURE_ROOT}" \
  --camera2-calibration \
    configs/sim_calibration_lower_pair.npz \
  --output "${TSDF_ROOT}" \
  --voxel-size 0.001 \
  --sdf-trunc 0.004 \
  --min-component-area 20 \
  --fill-kernel 5 \
  --fill-iterations 1

echo "============================================================"
echo "Stage 4: volume and topology evaluation"
echo "============================================================"

python -c "import trimesh,numpy as np; truth=1008.0; offset=float(np.load('configs/tsdf_volume_calibration.npz')['surface_offset_m']); full=trimesh.load_mesh('${TSDF_ROOT}/tsdf_mesh_raw.ply',process=False); components=list(full.split(only_watertight=False)); m=max(components,key=lambda x:len(x.faces)); trimesh.repair.fix_winding(m); trimesh.repair.fix_inversion(m,multibody=True); _,counts=np.unique(m.edges_sorted,axis=0,return_counts=True); raw=abs(m.volume)*1e6; corrected=(abs(m.volume)+offset*m.area)*1e6; section=m.section(plane_origin=[0.0,0.0,0.05],plane_normal=[1.0,0.0,0.0]); curves=[] if section is None else list(section.discrete); extents=[np.ptp(c[:,[1,2]],axis=0) for c in curves]; outer=any(np.all(np.abs(e-np.array([0.10,0.10]))<=np.array([0.015,0.015])) for e in extents); inner=any(np.all(np.abs(e-np.array([0.04,0.04]))<=np.array([0.012,0.012])) for e in extents); print('components=',len(components)); print('main_ratio=',len(m.faces)/len(full.faces)); print('boundary_edges=',np.sum(counts==1)); print('watertight=',m.is_watertight); print('surface_area_m2=',m.area); print('raw_cm3=',raw); print('raw_error_percent=',abs(raw-truth)/truth*100); print('corrected_cm3=',corrected); print('corrected_error_percent=',abs(corrected-truth)/truth*100); print('section_entities=',0 if section is None else len(section.entities)); print('closed_curve_count=',len(curves)); print('curve_extents_yz_m=',extents); print('outer_loop_found=',outer); print('inner_loop_found=',inner); print('hole_topology_pass=',inner)" \
  | tee "${TSDF_ROOT}/upper_lower_metrics.txt"

echo "COMPLETE $(date --iso-8601=seconds)" > "${STATUS_FILE}"

echo "Pipeline complete."
