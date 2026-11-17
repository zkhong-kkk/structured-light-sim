#!/usr/bin/env bash

set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CAPTURE_ROOT="$PROJECT/outputs/multiview_exr_fourcam"
RECON_ROOT="$PROJECT/outputs/multiview_reconstruction_exr_fourcam"

TSDF_OUTPUT="$PROJECT/outputs/tsdf_horizontal_hole_fourcam_12views"
POISSON_OUTPUT="$PROJECT/outputs/poisson_horizontal_hole_fourcam_direct"

STATUS="$PROJECT/outputs/fourcam_reconstruction_status.txt"

mkdir -p "$RECON_ROOT"
mkdir -p "$TSDF_OUTPUT"
mkdir -p "$POISSON_OUTPUT"
mkdir -p "$PROJECT/logs"

echo "RUNNING $(date -Iseconds)" > "$STATUS"

finish_status()
{
    code=$?

    if [ "$code" -ne 0 ]; then
        echo "FAILED $(date -Iseconds) code=$code" > "$STATUS"
    fi
}

trap finish_status EXIT

views=(
    0
    30
    60
    90
    120
    150
    180
    210
    240
    270
    300
    330
)

camera_ids=(
    upper_left
    upper_right
    lower_left
    lower_right
)

echo "============================================================"
echo "Stage 0: input validation"
echo "============================================================"

for camera_id in "${camera_ids[@]}"
do
    capture_dir="$CAPTURE_ROOT/$camera_id"
    calibration="$PROJECT/configs/sim_calibration_fourcam_${camera_id}.npz"

    if [ ! -d "$capture_dir" ]; then
        echo "ERROR: capture directory missing: $capture_dir"
        exit 1
    fi

    if [ ! -f "$calibration" ]; then
        echo "ERROR: calibration missing: $calibration"
        exit 1
    fi

    capture_count=$(find "$capture_dir" -name "0.hdf5" -type f | wc -l)

    echo "$camera_id captures: $capture_count / 144"

    if [ "$capture_count" -ne 144 ]; then
        echo "ERROR: incomplete captures for $camera_id"
        exit 1
    fi
done

echo "============================================================"
echo "Stage 1: reconstruct four camera point clouds"
echo "============================================================"

for camera_id in "${camera_ids[@]}"
do
    camera_recon="$RECON_ROOT/$camera_id"
    calibration="$PROJECT/configs/sim_calibration_fourcam_${camera_id}.npz"

    mkdir -p "$camera_recon"

    reconstructed_count=$(find "$camera_recon" -name "reconstruction_data.npz" -type f | wc -l)

    echo "$camera_id existing reconstructions: $reconstructed_count / 12"

    if [ "$reconstructed_count" -eq 12 ]; then
        echo "$camera_id reconstruction complete, skipping."
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Reconstructing camera: $camera_id"
    echo "------------------------------------------------------------"

    python reconstruction/batch_reconstruct_exr.py \
        --views "${views[@]}" \
        --input "$CAPTURE_ROOT/$camera_id" \
        --output "$camera_recon" \
        --calibration "$calibration"

    reconstructed_count=$(find "$camera_recon" -name "reconstruction_data.npz" -type f | wc -l)

    if [ "$reconstructed_count" -ne 12 ]; then
        echo "ERROR: $camera_id produced $reconstructed_count reconstructed views"
        exit 1
    fi

    echo "$camera_id reconstruction complete: 12 / 12"
done

total_reconstructed=$(find "$RECON_ROOT" -name "reconstruction_data.npz" -type f | wc -l)

echo "Total reconstructed views: $total_reconstructed / 48"

if [ "$total_reconstructed" -ne 48 ]; then
    echo "ERROR: expected 48 reconstructed views"
    exit 1
fi

echo "============================================================"
echo "Stage 2: four-camera TSDF fusion"
echo "============================================================"

python reconstruction/tsdf_fusion_multi_camera.py \
    --views "${views[@]}" \
    --rig upper_left \
        "$RECON_ROOT/upper_left" \
        "$CAPTURE_ROOT/upper_left" \
        "$PROJECT/configs/sim_calibration_fourcam_upper_left.npz" \
    --rig upper_right \
        "$RECON_ROOT/upper_right" \
        "$CAPTURE_ROOT/upper_right" \
        "$PROJECT/configs/sim_calibration_fourcam_upper_right.npz" \
    --rig lower_left \
        "$RECON_ROOT/lower_left" \
        "$CAPTURE_ROOT/lower_left" \
        "$PROJECT/configs/sim_calibration_fourcam_lower_left.npz" \
    --rig lower_right \
        "$RECON_ROOT/lower_right" \
        "$CAPTURE_ROOT/lower_right" \
        "$PROJECT/configs/sim_calibration_fourcam_lower_right.npz" \
    --output "$TSDF_OUTPUT" \
    --voxel-size 0.001 \
    --sdf-trunc 0.004 \
    --fill-kernel 5 \
    --fill-iterations 1 \
    --min-component-area 20 \
    --object-radius 0.09 \
    --min-z 0.003 \
    --max-z 0.115

if [ ! -f "$TSDF_OUTPUT/tsdf_mesh_raw.ply" ]; then
    echo "ERROR: TSDF mesh was not generated"
    exit 1
fi

echo "TSDF mesh generated:"
echo "$TSDF_OUTPUT/tsdf_mesh_raw.ply"

echo "============================================================"
echo "Stage 3: four-camera Poisson topology reconstruction"
echo "============================================================"

python reconstruction/poisson_from_multi_points.py \
    --views "${views[@]}" \
    --rig upper_left \
        "$RECON_ROOT/upper_left" \
        "$PROJECT/configs/sim_calibration_fourcam_upper_left.npz" \
    --rig upper_right \
        "$RECON_ROOT/upper_right" \
        "$PROJECT/configs/sim_calibration_fourcam_upper_right.npz" \
    --rig lower_left \
        "$RECON_ROOT/lower_left" \
        "$PROJECT/configs/sim_calibration_fourcam_lower_left.npz" \
    --rig lower_right \
        "$RECON_ROOT/lower_right" \
        "$PROJECT/configs/sim_calibration_fourcam_lower_right.npz" \
    --output "$POISSON_OUTPUT" \
    --object-radius 0.085 \
    --min-z 0.003 \
    --max-z 0.105 \
    --normal-radius 0.004 \
    --voxel-size 0.001 \
    --poisson-depth 10 \
    --poisson-scale 1.01

if [ ! -f "$POISSON_OUTPUT/poisson_mesh_raw.ply" ]; then
    echo "ERROR: Poisson mesh was not generated"
    exit 1
fi

echo "Poisson mesh generated:"
echo "$POISSON_OUTPUT/poisson_mesh_raw.ply"

echo "COMPLETE $(date -Iseconds) reconstructed_views=48" > "$STATUS"

echo "============================================================"
echo "Four-camera reconstruction pipeline complete"
echo "TSDF mesh:"
echo "$TSDF_OUTPUT/tsdf_mesh_raw.ply"
echo "Poisson mesh:"
echo "$POISSON_OUTPUT/poisson_mesh_raw.ply"
echo "============================================================"
