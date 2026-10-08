# VLM Inference Pipeline

Qwen3-VL (HuggingFace Transformers) 기반 caption 추론 파이프라인입니다.  
YAML config와 prompt 파일로 모델·입력·생성을 관리하고, 결과는 JSON으로 저장합니다.

## 요구 사항

- Python 3.10+
- NVIDIA GPU (이 환경: ARM64 GB10)
- 로컬 모델 (`configs/models.yaml` 프리셋):
  - `8b`: `/data/models/Qwen3-VL-8B-Instruct-BF16`
  - `8b-int8`: 같은 BF16 체크포인트를 bitsandbytes `load_in_8bit`으로 로드
  - `8b-fp8`: `/data/models/Qwen3-VL-8B-Instruct-FP8`
  - `30b`: `/data/models/Qwen3-VL-30B-A3B-Instruct` (MoE, 활성 3B)
  - `30b-fp8`: `/data/models/Qwen3-VL-30B-A3B-Instruct-FP8`

### 설치

이 레포에는 Python 3.12 가상환경 `.vlm-inf`가 준비되어 있습니다 (시스템 Python 3.14는 torch wheel 미지원).

```bash
cd ~/vlm-inference
source .vlm-inf/bin/activate

# 처음부터 다시 만들 때 (GB10 / CUDA 13):
# /usr/bin/python3.12 -m venv .vlm-inf
# source .vlm-inf/bin/activate
# pip install -U pip
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# pip install "transformers>=4.57.0" accelerate "bitsandbytes>=0.50.1" qwen-vl-utils PyYAML Pillow av
```

`transformers>=4.57`이 Qwen3-VL에 필요합니다.

## 빠른 실행

1. [`configs/default.yaml`](configs/default.yaml)의 `jobs`에 실제 경로를 넣습니다.
2. 필요하면 [`prompts/`](prompts/)의 텍스트를 수정하거나 새 파일을 만듭니다.
3. 실행:

```bash
python run_inference.py --config configs/default.yaml
```

8B와 30B는 같은 실험 config에서 바꿔 끼울 수 있습니다. YAML은 `model: 8b` 또는 `model: 30b`이고, `--model`이 config 값을 덮어씁니다.

```bash
# 30B MoE (/data/models/Qwen3-VL-30B-A3B-Instruct)
.vlm-inf/bin/python run_inference.py \
  --config configs/default.yaml \
  --model 30b \
  --output-dir outputs/default_30b

# 같은 jobs를 8B로
.vlm-inf/bin/python run_inference.py \
  --config configs/default.yaml \
  --model 8b

# low-camera-motion caption (30B면 출력은 *_30b 로 분리)
.vlm-inf/bin/python scripts/run_low_camera_motion_captioning.py --model 30b
```

결과는 `outputs/results_MMDDHHmmss.json`에 저장됩니다 (실행 시각 타임스탬프). `--output-dir`을 주면 feasibility cache도 그 아래로 옮깁니다.

## Inference mode

`configs/default.yaml`의 `mode`:

| mode | 동작 |
|------|------|
| `default` | `image_dir` 프레임을 video sequence로 묶어 입력 |
| `experiment` | 파일명에서 Global/ROI·시간을 파싱해 `[Global] t=12.0s` + 이미지 형태로 교차 입력 |

experiment 파일명 예: `global_frame_000004_t12.0.jpg`, `tube01_frame_000007_t3.0.jpg`  
패턴/라벨은 `experiment:` 섹션의 regex·format으로 바꿀 수 있습니다.

## 입력 모드 (`jobs[].type`)

| type | 설명 | 샘플 단위 |
|------|------|-----------|
| `video` | 단일 video 파일 | 파일 1개 |
| `video_dir` | video들이 들어있는 폴더 | video 파일 1개 |
| `images` | image 경로 리스트 (`paths`) | 리스트 전체 (멀티이미지 1샘플) |
| `image_dir` | frame/image 폴더 | 폴더 전체 (frame sequence) |
| `image_dirs` | 하위 폴더 여러 개 | 하위 폴더 1개 |

## Prompt

우선순위: **샘플 sidecar > job `prompt` > job `prompt_file` > 전역 `prompt` > 전역 `prompt_file`**

- 인라인: `prompt: "Describe ..."`
- 파일: `prompt_file: prompts/detailed.txt`
- 샘플별 교체: 입력 옆 `{sample_id}.prompt.txt`  
  예: `clip_001.mp4` → `clip_001.prompt.txt`

## Config 요약

```yaml
model: 8b   # 8b | 8b-int8 | 30b | 8b-fp8 | 30b-fp8  (configs/models.yaml)
# 또는 경로 매핑 / name 오버레이:
# model:
#   name: 30b
#   device_map: auto

generation:
  max_new_tokens: 512
  do_sample: false

video:
  fps: 2.0
  min_pixels: 4096
  max_pixels: 262144
  total_pixels: 20971520

prompt_file: prompts/caption_default.txt

output:
  dir: outputs
  file: results.json

jobs:
  - name: my_videos
    type: video_dir
    path: /data/datasets/.../videos
    glob: "*.mp4"
    prompt_file: prompts/detailed.txt
```

## 출력 JSON

```json
[
  {
    "job": "my_videos",
    "sample_id": "clip_001",
    "input_type": "video",
    "paths": ["file:///data/.../clip_001.mp4"],
    "prompt": "...",
    "response": "...",
    "model": "/data/models/Qwen3-VL-8B-Instruct-BF16",
    "error": null
  }
]
```

실패한 샘플은 `error`에 메시지가 들어가고 이어서 진행합니다. job마다 JSON을 flush합니다.

## Caption 추출

Results JSON에서 caption만 뽑을 때:

```bash
python extract_feasibility_captions.py \
  outputs/feasibility_results_roi4_0903124216.json
```

(`full_2fps`는 스크립트에서 자동 제외됩니다.)

## 프로젝트 구조

```
configs/default.yaml
prompts/
src/vlm_inference/
  config.py
  inputs.py
  prompts.py
  model.py
  runner.py
run_inference.py
```

ROI 샘플링 속도를 모션에 맞춰 바꾸던 실험의 조건과 프레임 수는
`docs/motion_adaptive_roi_sampling.md`에 있다.

## VIRAT dense-caption GT candidate pilot

대표 5개 원본 영상을 Qwen3-VL-30B-A3B-Instruct로 전체 시간 범위에서 분석하고, event마다 `timestamp`와 `sentence`만 저장합니다. 실행 명령·출력·prompt·실제 프레임/fps는 [pilot 문서](docs/virat_dense_caption_pilot.md)를 참고하세요.

## VIRAT 3-second clip-caption pilot

기존 대표 5개를 연속된 실제 3초 video segment로 나누고 각 구간을 독립적으로 Qwen3-VL-30B-A3B-Instruct에 입력합니다. 마지막 짧은 구간을 포함하며, 시간 경계는 segmentation 코드가 기록합니다. caption merge나 최종 GT 생성 없이 원본 영상별 JSON/TXT를 저장합니다. 실행 명령·출력·sampling 및 생성 상태는 [clip pilot 문서](docs/virat_clip_caption_pilot.md)를 참고하세요.

## VIRAT text-only temporal event merge pilot

기존 3초 clip caption JSON 하나를 text input으로 Qwen에 전달해 시간적으로 연속된 event candidate를 병합합니다. `source_clips`와 원본 segmentation 경계를 보존하며 원본 영상 inference는 실행하지 않습니다. 원본 caption과 별도의 JSON/MD에 저장합니다. [실행 및 provenance 검증](docs/virat_temporal_event_merge.md)을 참고하세요.

## Selected-59 VIRAT clip caption → temporal merge

선정된 59개를 총 1,100개 3초 clip으로 caption한 뒤, 모든 caption이 정상 완료되면 text-only aggressive temporal merge를 실행합니다. 같은 프로세스의 Qwen30B model을 재사용하고 `--resume`으로 정상 결과를 건너뜁니다. [실행·재개·진행 상태 문서](docs/virat_clip_caption_pipeline.md)를 참고하세요.

## evaluation/

기존 `vlm-evaluation/` 전체를 `evaluation/`으로 옮겼습니다. 하위 구조는 `configs/`(설정), `scripts/`(실행·검증), `vantage_vqa/`(평가 코드), `experiments/`(GT·실험 기록)입니다.

실행 환경은 프로젝트 루트의 `.vlm-inf`입니다. Annotation UI를 사용하려면 `.vlm-inf/bin/python -m pip install -r evaluation/requirements-annotation-ui.txt`로 추가 의존성을 설치합니다.

사람이 검수한 GT는 `evaluation/experiments/virat/caption_event_gt_191/gt/`와 `gt_human_review_v2/`에 있으며, 현재 검수본은 같은 디렉터리의 `CURRENT_REVIEWED_GT.json`을 참고합니다. 추가 GT·실험 JSON은 `evaluation/experiments/virat/caption_event_gt_191_v5/`와 `caption_event_pilot_6/`에 있습니다.
