#!/usr/bin/env bash
# Full pipeline: motion-detector Stage1–3 + VLM ROI-start/centroid captioning.
set -euo pipefail

MD_ROOT=/home/ssandree/motion-detector
VLM_ROOT=/home/ssandree/vlm-inference
LOG_DIR=${VLM_ROOT}/outputs/low_camera_motion_start_centroid
mkdir -p "${LOG_DIR}"
LOG=${LOG_DIR}/pipeline.log

source /home/ssandree/.local/opencv-cuda/env.sh
export PYTHONPATH="${MD_ROOT}/src:${PYTHONPATH:-}"

echo "[$(date -Is)] motion-detector start" | tee -a "${LOG}"
/home/ssandree/miniconda3/bin/python3.14 \
  "${MD_ROOT}/scripts/run_low_camera_motion_clips.py" \
  --wait_stable_sec 60 \
  --wait_stable_max_sec 1800 \
  "$@" | tee -a "${LOG}"

# Drop CUDA OpenCV PYTHONPATH so the VLM venv's cv2 loads.
unset PYTHONPATH
unset OPENCV_CUDA_PREFIX

echo "[$(date -Is)] VLM captioning start" | tee -a "${LOG}"
"${VLM_ROOT}/.vlm-inf/bin/python" \
  "${VLM_ROOT}/scripts/run_low_camera_motion_captioning.py" \
  | tee -a "${LOG}"

echo "[$(date -Is)] done" | tee -a "${LOG}"
echo "captions: ${LOG_DIR}/captions.md"
