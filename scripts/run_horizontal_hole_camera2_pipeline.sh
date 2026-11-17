#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CAPTURE_ROOT="outputs/multiview_exr_horizontal_hole_camera2_12views"
RECON_ROOT="outputs/multiview_reconstruction_exr_horizontal_hole_camera2_12views"
TSDF_ROOT="outputs/tsdf_horizontal_hole_camera2_12views"
SMOKE_ROOT="outputs/horizontal_hole_camera2_smoke_120"
STATUS_FILE="outputs/horizontal_hole_camera2_pipeline_status.txt"

VIEWS=(
  0 30 60 90 120 150
  180 210 240 270 300 330
)

cd "${PROJECT_ROOT}"
if [[ -n "${STRUCTURED_LIGHT_VENV:-}" ]]; then
  source "${STRUCTURED_LIGHT_VENV}/bin/activate"
fi
export CUDA_VISIBLE_DEVICES=0

mkdir -p outputs

echo "RUNNING $(date --iso-8601=seconds)" > "${STATUS_FILE}"

on_error() {
  code=$?
  echo "FAILED $(date --iso-8601=seconds) line=${BASH_LINENO[0]} code=${code}" \
    > "${STATUS_FILE}"
  exit "${code}"
}

trap on_error ERR

echo "============================================================"
echo "Stage 1: camera2 smoke render"
echo "============================================================"

if [ ! -f "${SMOKE_ROOT}/0.hdf5" ]; then
  blenderproc run \
    simulator/single_phase_capture_horizontal_hole_camera2.py \
    --periods 12 \
    --phase-degree 0 \
    --turntable-angle 120 \
    --output "${SMOKE_ROOT}"
else
  echo "Smoke HDF5 already exists, skipping render."
fi

python -c "import h5py,numpy as np,cv2; f=h5py.File('${SMOKE_ROOT}/0.hdf5','r'); x=f['colors'][:,:,:3].astype(np.float32); scale=np.percentile(x,99.5); x=np.clip(x/max(scale,1e-8),0,1)**(1/2.2); cv2.imwrite('${SMOKE_ROOT}/color_preview.png',(x[:,:,::-1]*255).astype(np.uint8)); print('smoke preview saved')"

echo "============================================================"
echo "Stage 2: render 12 views, expected 144 HDF5 files"
echo "============================================================"

python simulator/batch_render_horizontal_hole_camera2.py \
  --gpu 0 \
  --views "${VIEWS[@]}" \
  --output "${CAPTURE_ROOT}"

capture_count=$(
  find "${CAPTURE_ROOT}" -name "0.hdf5" | wc -l
)

echo "capture_count=${capture_count}"

if [ "${capture_count}" -ne 144 ]; then
  echo "Expected 144 captures, found ${capture_count}" >&2
  exit 20
fi

echo "============================================================"
echo "Stage 3: decode and reconstruct 12 views"
echo "============================================================"

python reconstruction/batch_reconstruct_exr.py \
  --views "${VIEWS[@]}" \
  --input "${CAPTURE_ROOT}" \
  --output "${RECON_ROOT}" \
  --calibration configs/sim_calibration_camera2.npz

reconstruction_count=$(
  find "${RECON_ROOT}" \
    -name reconstruction_data.npz | wc -l
)

echo "reconstruction_count=${reconstruction_count}"

if [ "${reconstruction_count}" -ne 12 ]; then
  echo "Expected 12 reconstructions, found ${reconstruction_count}" >&2
  exit 30
fi

echo "============================================================"
echo "Stage 4: camera2 TSDF"
echo "============================================================"

python reconstruction/tsdf_fusion.py \
  --views "${VIEWS[@]}" \
  --reconstruction-root "${RECON_ROOT}" \
  --capture-root "${CAPTURE_ROOT}" \
  --calibration configs/sim_calibration_camera2.npz \
  --output "${TSDF_ROOT}" \
  --voxel-size 0.001 \
  --sdf-trunc 0.004 \
  --fill-kernel 5 \
  --fill-iterations 1

test -f "${TSDF_ROOT}/tsdf_mesh_raw.ply"

echo "============================================================"
echo "Stage 5: volume metrics"
echo "============================================================"

python -c "import trimesh,numpy as np; truth=1008.0; offset=float(np.load('configs/tsdf_volume_calibration.npz')['surface_offset_m']); full=trimesh.load_mesh('${TSDF_ROOT}/tsdf_mesh_raw.ply',process=False); components=list(full.split(only_watertight=False)); m=max(components,key=lambda x:len(x.faces)); trimesh.repair.fix_winding(m); trimesh.repair.fix_inversion(m,multibody=True); _,counts=np.unique(m.edges_sorted,axis=0,return_counts=True); raw=abs(m.volume)*1e6; corrected=(abs(m.volume)+offset*m.area)*1e6; print('components=',len(components)); print('main_ratio=',len(m.faces)/len(full.faces)); print('area_m2=',m.area); print('boundary_edges=',np.sum(counts==1)); print('raw_cm3=',raw); print('raw_error_percent=',abs(raw-truth)/truth*100); print('corrected_cm3=',corrected); print('corrected_error_percent=',abs(corrected-truth)/truth*100)" \
  > "${TSDF_ROOT}/camera2_volume_metrics.txt"

cat "${TSDF_ROOT}/camera2_volume_metrics.txt"

echo "============================================================"
echo "Stage 6: topology section check"
echo "============================================================"

python -c "import trimesh,numpy as np; full=trimesh.load_mesh('${TSDF_ROOT}/tsdf_mesh_raw.ply',process=False); m=max(full.split(only_watertight=False),key=lambda x:len(x.faces)); tx=m.triangles[:,:,0]; crossing=(tx.min(axis=1)<=0.0)&(tx.max(axis=1)>=0.0)&((tx.max(axis=1)-tx.min(axis=1))>1e-9); section=m.section(plane_origin=[0.0,0.0,0.05],plane_normal=[1.0,0.0,0.0]); curves=[] if section is None else list(section.discrete); curves=sorted(curves,key=lambda c:np.prod(np.ptp(c[:,[1,2]],axis=0)),reverse=True); print('triangles_crossing_x0=',crossing.sum()); print('section_is_none=',section is None); print('section_entities=',0 if section is None else len(section.entities)); print('closed_curve_count=',len(curves)); [print(i,'extent_yz_m=',np.ptp(c[:,[1,2]],axis=0),'closed_gap_m=',np.linalg.norm(c[0]-c[-1])) for i,c in enumerate(curves[:10])]" \
  > "${TSDF_ROOT}/camera2_topology_metrics.txt"

cat "${TSDF_ROOT}/camera2_topology_metrics.txt"

echo "COMPLETE $(date --iso-8601=seconds)" > "${STATUS_FILE}"

echo "============================================================"
echo "Pipeline complete"
echo "Status: ${STATUS_FILE}"
echo "Volume: ${TSDF_ROOT}/camera2_volume_metrics.txt"
echo "Topology: ${TSDF_ROOT}/camera2_topology_metrics.txt"
echo "============================================================"
