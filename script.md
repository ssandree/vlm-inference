저장소 루트에서 `.vlm-inf/bin/python`으로 실행한다.

모델 프리셋은 `configs/models.yaml`에 있다. `8b`, `8b-int8`, `8b-fp8`, `30b`, `30b-fp8`이다. 실험 YAML의 `model`을 바꾸거나, 아래처럼 `--model`로 덮어쓴다.

```bash
.vlm-inf/bin/python run_inference.py \
  --config configs/default.yaml \
  --model 30b \
  --output-dir outputs/default_30b
```

## 1. 일반 VLM caption

진입점은 `run_inference.py`다.

```bash
.vlm-inf/bin/python run_inference.py \
  --config configs/default.yaml
```

`configs/default.yaml`의 `jobs`에 지정된 video, image directory, image list를 읽어 Qwen3-VL caption을 만든다.

지원하는 입력 타입:

- `video`: video 파일 하나
- `video_dir`: 디렉터리 안 video 전체
- `images`: 이미지 여러 장을 하나의 sample로 입력
- `image_dir`: 한 폴더의 frame sequence
- `image_dirs`: 하위 폴더를 각각 하나의 sample로 처리

결과는 `output.dir` 아래 타임스탬프가 붙은 JSON이다. `--output-dir`을 주면 feasibility cache도 그 아래로 옮긴다.

공통 옵션:

```text
--config CONFIG
--model SPEC
--output-dir DIR
--log-level DEBUG|INFO|WARNING|ERROR
--input-order timestamp|grouped|global_anchor|tube_story
--roi-frames default|fps_1
--global-insert video_start|start_only|start_end|start_mid_end
--global-insert-source fps_1|fps_2|fps_0_25
--roi-repr separate_roi|union_roi
--roi-sampling spatial_occupancy|piecewise_fps|fixed_1fps|fixed_2fps|adaptive_0.5_1_2|adaptive_0.5_1_4|accumulated_motion|accumulated_motion_cap2fps|burst_2fps|centroid_2fps
--roi-sampling-tau FLOAT
--overwrite-roi-sampling
--overwrite-roi-repr
--save-vlm-inputs
--no-resume
--dry-run
--prepare-single-motion
```

`--dry-run`과 `--prepare-single-motion`은 feasibility config에서만 쓴다. ROI 샘플링 속도를 모션에 맞춰 바꾸던 실험의 조건과 프레임 수는 `docs/motion_adaptive_roi_sampling.md`에 있다. 그 전용 실행 스크립트와 산출물은 없고, 샘플러는 아래 Global×ROI 실험이 그대로 호출한다.

## 2. Feasibility

`configs/feasibility.yaml`은 VIRAT 9개 영상에서 full 1 fps, Global 0.25 fps + proposed multi-ROI, single-motion ROI를 비교한다. 실행은 `src/vlm_inference/runner.py`의 feasibility 경로로 들어간다.

```bash
.vlm-inf/bin/python run_inference.py \
  --config configs/feasibility.yaml
```

흐름은 ROI/sampling cache 준비, method별 입력 frame 수집, cache hit 확인, caption 생성, video/method별 JSON 저장이다.

cache를 무시하고 다시 돌릴 때:

```bash
.vlm-inf/bin/python run_inference.py \
  --config configs/feasibility.yaml \
  --no-resume
```

모델을 올리지 않고 frame 수와 timestamp만 볼 때:

```bash
.vlm-inf/bin/python run_inference.py \
  --config configs/feasibility.yaml \
  --dry-run
```

## 3. Single-motion ROI 준비

proposed motion tube를 합쳐 single-motion ROI cache를 만든다. VLM은 로드하지 않는다.

```bash
.vlm-inf/bin/python run_inference.py \
  --config configs/feasibility.yaml \
  --prepare-single-motion
```

## 4. Full-frame vs Motion Multi-ROI

`scripts/run_fullframe_motion_multiroi.py`가 full-frame과 Motion ROI 조건을 모델 크기별로 돌린다. 기본 설정은 `configs/fullframe_motion_multiroi.yaml`이다. 11개 clip × 5개 프리셋 × `full_frame` / `motion_roi`는 `configs/fullframe_proposed_roi_5model.yaml`이다.

```bash
.vlm-inf/bin/python scripts/run_fullframe_motion_multiroi.py --dry-run

.vlm-inf/bin/python scripts/run_fullframe_motion_multiroi.py \
  --config configs/fullframe_proposed_roi_5model.yaml \
  --models 8b 30b \
  --conditions full_frame motion_roi
```

조건 이름은 `full_frame`, `motion_roi`, `motion_multi_roi`, `motion_multi_roi_start_anchor`, `motion_multi_roi_peak_anchor`다.

## 5. Global frame 삽입

ROI crop은 두고 Global 시각만 바꾼다. `scripts/run_global_frame_insert.py`의 기본은 dry-run이고, caption은 `--inference`가 켠다. 조건은 `video_start`, `start_only`, `start_end`다.

```bash
.vlm-inf/bin/python scripts/run_global_frame_insert.py \
  --config configs/feasibility_global_frame.yaml \
  --output-dir outputs/global_frame_insert

.vlm-inf/bin/python scripts/run_global_frame_insert.py --inference
```

같은 축을 `run_inference.py --global-insert`로 고를 수도 있다. `configs/feasibility_global_insert.yaml`은 ROI 시작·끝·중간(`start_only`, `start_end`, `start_mid_end`)이다.

Global 삽입 두 가지(`video_start`, `start_only`)와 ROI 샘플러 네 가지(고정 1 fps, 고정 2 fps, 누적 τ=8, centroid 2 fps)의 교차는 `configs/feasibility_global_roi_cross.yaml`이다. 비교 표는 추론 결과 JSON이 `outputs/global_roi_cross/`에 있을 때 다음이 만든다.

```bash
.vlm-inf/bin/python run_inference.py \
  --config configs/feasibility_global_roi_cross.yaml

.vlm-inf/bin/python scripts/export_global_roi_cross.py
```

## 6. Global resolution ablation

ROI 샘플링과 Global 시각은 고정하고 Global 공간 해상도만 `1.0`, `0.5`, `0.25`, `0.125`로 바꾼다. 기본은 dry-run이고, caption은 `--inference`다.

```bash
.vlm-inf/bin/python scripts/run_global_resolution_ablation.py \
  --config configs/feasibility_global_resolution_ablation.yaml \
  --output-dir outputs/global_resolution_ablation

.vlm-inf/bin/python scripts/run_global_resolution_ablation.py --inference
```

## 7. Separate ROI와 Union ROI

`scripts/run_roi_union_repr.py`는 tube를 따로 둘지, 하나의 union crop으로 합칠지 준비한다. caption은 이후 `run_inference.py`가 만든다.

```bash
.vlm-inf/bin/python scripts/run_roi_union_repr.py \
  --config configs/feasibility_roi_union.yaml

.vlm-inf/bin/python scripts/run_roi_union_repr.py \
  --config configs/feasibility_roi_union.yaml \
  --roi-repr separate_roi union_roi \
  --overwrite
```

## 8. Centroid ROI caption

저카메라 모션 목록을 읽어 ROI 시작 Global과 centroid-displacement ROI로 caption한다. 기본 모델은 `8b`이고, 30B는 `--model 30b`다.

```bash
.vlm-inf/bin/python scripts/run_low_camera_motion_captioning.py --dry-run
.vlm-inf/bin/python scripts/run_low_camera_motion_captioning.py --model 30b
```

sample30은 같은 입력 구성의 feasibility config다.

```bash
.vlm-inf/bin/python run_inference.py \
  --config configs/sample30_30b_start_centroid.yaml
```

저장된 centroid 쌍과 고정 padding 비용만 보는 진단은 추론을 하지 않는다.

```bash
.vlm-inf/bin/python scripts/analyze_centroid_displacement.py \
  --config configs/centroid_displacement_analysis.yaml
```

ROI 시작 박스 스크리닝이 기존 입력과 같은지는 `scripts/verify_roi_start_box.py`가 `configs/feasibility_global_roi_cross.yaml`과 `configs/feasibility_global_roi_start_box.yaml`을 비교한다. 인자 없이 실행한다.

## 9. Visual information

`scripts/analyze_token_information.py`는 caption 파이프라인이 아니다. 독립 full-frame을 64×64 block으로 나누고, GT event 문장 log probability와 block 제거/유지로 ROI별 information coverage, heatmap, JSON/CSV를 만든다. 프레임은 한 영상에서 5장 이상 10장 이하여야 한다. 현재 `configs/token_information_feasibility.yaml`은 `configs/token_information_frame_event_map.yaml`의 매칭으로 8장을 지정한다.

```bash
.vlm-inf/bin/python scripts/analyze_token_information.py \
  --config configs/token_information_feasibility.yaml \
  --layers 0 4 8 12 \
  --output-dir outputs/token_information_feasibility
```

옵션은 `--config`, `--layers`, `--batch-size`, `--output-dir`, `--map`이다.

영상 전체 프레임을 한 multimodal 입력으로 두고 layer-4 visual information volume을 계산할 때:

```bash
.vlm-inf/bin/python scripts/analyze_video_information_volume.py \
  --config configs/video_information_volume.yaml

.vlm-inf/bin/python scripts/analyze_video_information_volume.py \
  --config configs/video_information_volume_9videos.yaml
```

그 volume과 기존 VIRAT ROI tube를 비교할 때:

```bash
.vlm-inf/bin/python scripts/analyze_vlm_roi_alignment.py \
  --config configs/video_information_volume_9videos.yaml \
  --output-root outputs/vlm_roi_alignment
```

`--render-only`는 저장된 volume으로 그림만 다시 그린다.

## 10. Caption 추출

`extract_feasibility_captions.py`는 feasibility 결과 JSON에서 video별 caption만 추린다. `full_2fps`는 빼 둔다.

```bash
.vlm-inf/bin/python extract_feasibility_captions.py \
  outputs/feasibility_results_roi4_0904030825.json \
  -o outputs/my_captions.json \
  --methods full_1fps global_0.25fps_proposed_multi_roi
```

caption이 없는 method도 남기려면 `--include-null`을 붙인다.

`extract_job_captions.py`는 `job`과 `caption`만 뽑는다. 출력 기본값은 입력 옆 `<stem>_job_captions.json`이다.

```bash
.vlm-inf/bin/python extract_job_captions.py \
  outputs/feasibility_results_roi4_0904030825.json
```

## 11. VLM 입력 board

`viz_vlm_input_boards.py`는 모델을 다시 실행하지 않고 `outputs/input_order/`의 dump를 읽어 입력 frame 순서 board를 만든다.

```bash
.vlm-inf/bin/python viz_vlm_input_boards.py \
  --video_id VIRAT_S_000201_06_001354_001397 \
  --layout both
```

주요 옵션은 `--dump_dir`, `--output_root`, `--video_id`, `--method`, `--input_order`, `--layout grouped|order|both`다. 기본 method는 `full_1fps`, `global_0.25fps_proposed_multi_roi`, `single_motion_roi_original`, `global_0.25fps_single_motion_roi`다.

## Config

| Config | 용도 |
| --- | --- |
| `default.yaml` | 일반 image/video caption |
| `models.yaml` | `8b`, `8b-int8`, `8b-fp8`, `30b`, `30b-fp8` 가중치 경로 |
| `feasibility.yaml` | full 1 fps, Global 0.25 fps + proposed ROI, single-motion ROI |
| `feasibility_global_frame.yaml` | Global을 video 시작 / ROI 시작 / ROI 시작·끝에 두는 3조건 |
| `feasibility_global_insert.yaml` | ROI 시작, 시작·끝, 시작·중간·끝 |
| `feasibility_global_first.yaml` | 0.25 fps Global 첫 장 + ROI 1 fps |
| `feasibility_global_roi_cross.yaml` | Global 2조건 × ROI 샘플러 4조건, 9개 영상 |
| `feasibility_global_roi_cross_probe.yaml` | 위 교차의 짧은 probe |
| `feasibility_global_roi_start_box.yaml` | ROI 시작 박스 + centroid 2 fps / 고정 2 fps |
| `feasibility_global_roi_video_start_centroid_pad1.yaml` | video 시작 centroid, `pad_sec=1.0` |
| `feasibility_global_roi_video_start_centroid_pad2.yaml` | video 시작 centroid, `pad_sec=2.0` |
| `feasibility_global_resolution_ablation.yaml` | Global 해상도만 변경 |
| `feasibility_roi_union.yaml` | separate ROI vs union ROI |
| `feasibility_roi_wise.yaml` | start-only Global과 tube별 merge |
| `feasibility_start_only_prompt.yaml` | start-only 입력에서 prompt만 변경 |
| `feasibility_start_only_roi.yaml` | start-only Global + 고정 2 fps / 누적 모션 ROI |
| `feasibility_full1fps_video_probe.yaml` | 1 fps JPEG를 still이 아니라 video clip으로 |
| `feasibility_sampling_probe.yaml` | 두 clip의 sampling 입력 확인 |
| `feasibility_tight_probe.yaml` | 한 clip, per-frame cell AABB ROI |
| `fullframe_motion_multiroi.yaml` | full-frame vs Motion Multi-ROI |
| `fullframe_proposed_roi_5model.yaml` | 11 clip × 5 모델 × full-frame / motion ROI |
| `low_camera_motion_start_centroid.yaml` | 저카메라 모션 caption이 쓰는 feasibility YAML |
| `sample30_30b_start_centroid.yaml` | sample30, 30B, centroid ROI, start-only Global |
| `centroid_displacement_analysis.yaml` | 저장된 centroid 진단. 추론 없음 |
| `token_information_feasibility.yaml` | full-frame block information |
| `token_information_frame_event_map.yaml` | token-information의 frame↔event 매칭 |
| `video_information_volume.yaml` | 한 영상의 layer-4 information volume |
| `video_information_volume_9videos.yaml` | 9개 영상의 information volume |

## 테스트

```text
tests/test_centroid_motion.py
tests/test_fullframe_precision_sweep.py
tests/test_global_insert.py
tests/test_global_resolution_ablation.py
tests/test_model_catalog.py
tests/test_token_information.py
```

```bash
.vlm-inf/bin/python -m pytest -q
.vlm-inf/bin/python -m pytest -q tests/test_token_information.py
.vlm-inf/bin/python -m py_compile run_inference.py scripts/analyze_token_information.py
```

dry-run, centroid 진단, caption 추출, 입력 board, `export_global_roi_cross.py`, `verify_roi_start_box.py`는 모델을 로드하지 않는다.
