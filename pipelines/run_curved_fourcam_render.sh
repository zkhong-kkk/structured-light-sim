#!/usr/bin/env bash

set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

OUTPUT="$PROJECT/outputs/multiview_exr_curved_fourcam"
STATUS="$PROJECT/outputs/curved_fourcam_render_status.txt"

mkdir -p "$OUTPUT"
mkdir -p "$PROJECT/logs"

echo "RUNNING $(date -Iseconds)" > "$STATUS"

trap '
code=$?
if [ "$code" -ne 0 ]; then
    echo "FAILED $(date -Iseconds) code=$code" > "$STATUS"
fi
' EXIT

camera_ids=(
    upper_left
    upper_right
    lower_left
    lower_right
)

views=(
    0 30 60 90 120 150
    180 210 240 270 300 330
)

frequencies=(1 4 12)
phases=(0 90 180 270)

total=576
completed=0

for camera_id in "${camera_ids[@]}"
do
    for view in "${views[@]}"
    do
        for frequency in "${frequencies[@]}"
        do
            for phase in "${phases[@]}"
            do
                completed=$((completed + 1))

                view_name=$(printf "%03d" "$view")
                frequency_name=$(printf "%02d" "$frequency")
                phase_name=$(printf "%03d" "$phase")

                output_dir="$OUTPUT/$camera_id/view_$view_name/freq_$frequency_name/phase_$phase_name"

                echo "============================================================"
                echo "[$completed/$total] camera=$camera_id view=$view frequency=$frequency phase=$phase"
                echo "output=$output_dir"
                echo "time=$(date -Iseconds)"

                if [ -f "$output_dir/0.hdf5" ]; then
                    echo "HDF5 exists, skipping."
                    continue
                fi

                mkdir -p "$output_dir"

                CUDA_DEVICE_ORDER=PCI_BUS_ID \
                CUDA_VISIBLE_DEVICES="$GPU_ID" \
                blenderproc run \
                    simulator/single_phase_capture_curved_fourcam.py \
                    --camera-id "$camera_id" \
                    --periods "$frequency" \
                    --phase-degree "$phase" \
                    --turntable-angle "$view" \
                    --output "$output_dir"

                if [ ! -f "$output_dir/0.hdf5" ]; then
                    echo "ERROR: missing $output_dir/0.hdf5"
                    exit 1
                fi
            done
        done
    done
done

count=$(find "$OUTPUT" -name "0.hdf5" -type f | wc -l)

if [ "$count" -ne 576 ]; then
    echo "ERROR: expected 576 HDF5 files, found $count"
    exit 1
fi

echo "COMPLETE $(date -Iseconds) captures=$count" > "$STATUS"
echo "Four-camera rendering complete: $count files"
