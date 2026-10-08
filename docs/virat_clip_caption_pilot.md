# VIRAT 3-second independent clip-caption pilot

기존 full-video pilot에서 선택한 대표 5개만 처리한다. 각 원본을 연속적인 `[0,3)`, `[3,6)`, … 구간으로 나누고 마지막 짧은 구간도 포함한다. 모델에 timestamp 예측, event taxonomy, structured schema를 요청하지 않는다. 인접 caption merge, 중복 제거, 최종 GT 생성은 하지 않는다.

## 실행

```bash
cd /home/ssandree/Motion-guided-ROI/vlm-inference

# 실제 segment 생성 및 모든 processor 입력 검증, 모델 가중치 로딩 없음
.vlm-inf/bin/python scripts/run_virat_clip_caption_pilot.py --dry-run

# 대표 5개, 총 51개 clip에 대한 실제 추론
.vlm-inf/bin/python scripts/run_virat_clip_caption_pilot.py

# 중단된 실행 이어서 진행: 동일 설정/source의 완료 clip만 재사용
.vlm-inf/bin/python scripts/run_virat_clip_caption_pilot.py --resume

# prompt/settings 변경 시 별도의 출력 폴더
.vlm-inf/bin/python scripts/run_virat_clip_caption_pilot.py \
  --output-dir outputs/virat_clip_caption_pilot_rerun
```

동일 destination에서 dry-run 후 inference 실행이 가능하다. 동일 종류의 실행에 기존 기록이 있으면 `--resume`을 사용한다. resume은 clip별 inference 기록을 재사용해 영상별 JSON/TXT를 다시 만든다. 사람이 편집한 JSON/TXT는 별도로 보관한다. 설정/source fingerprint가 다르면 새 출력 폴더를 요구한다. 같은 폴더의 동시 실행은 lock으로 차단한다. 빈 응답·토큰 한도 도달은 정상 응답으로 처리하지 않으며, 원문과 상태를 보존한다. resume은 이런 생성 기록도 유지하고 남은 clip을 진행한다. 해당 clip을 다시 생성하려면 새 output directory를 사용한다. 모든 clip 처리가 끝나도 빈 응답/잘림이 있으면 종료 코드는 1이다.

## 영상과 경계

| Video | Clip 수 | 실제 길이 (s) |
|---|---:|---:|
| VIRAT_S_010001_03_000537_000563.mp4 | 7 | 20.692532 |
| VIRAT_S_010106_05_000954_000996.mp4 | 10 | 27.409262 |
| VIRAT_S_010113_05_000776_000805.mp4 | 8 | 22.778473 |
| VIRAT_S_010208_09_000857_000886.mp4 | 7 | 20.358782 |
| VIRAT_S_040005_00_000003_000058.mp4 | 19 | 54.721333 |

길이는 기존 `inspect_video`와 동일하게 native frame count / fps로 계산한다. filename의 숫자를 시간으로 해석하지 않는다. 모델 응답에서 시간 경계를 추출하지 않는다.

프레임의 native timestamp가 `[start_sec,end_sec)`에 속하면 해당 segment에 포함한다. 경계는 `ceil(start_sec * native_fps)` / `ceil(end_sec * native_fps)`를 사용하므로 연속 clip에 프레임이 겹치거나 빠지지 않는다. 비정수 fps에서는 frame grid와 3초 경계 사이에 최대 한 프레임 간격의 차이가 있다. 원본 시간 경계와 source frame 범위를 모두 기록한다.

FFmpeg `trim`으로 각 구간의 **모든 원본 프레임**을 별도 FFV1 lossless AVI에 저장하고 timestamp를 segment 시작 기준으로 초기화한다. 그 video 파일 하나씩을 독립 user message로 Qwen에 입력한다. 이전 clip 응답/GT/ROI 정보는 전달하지 않는다. 모델은 재사용하되 입력과 생성 context는 clip마다 새로 만든다.

## 모델과 sampling

- Config: `configs/virat_clip_caption_pilot.yaml`, preset `30b`.
- Model: `/data/models/Qwen3-VL-30B-A3B-Instruct`, BF16, `device_map:auto`, SDPA; INT8/FP8 대체 없음.
- 기존 `Qwen3VLCaptioner.generate`와 `prepare_multimodal_inputs`, `_apply_video_opts`, PyAV reader를 재사용한다.
- 요청 video fps: **2.0**. 각 clip 전체에서 Qwen `smart_nframes` 후 `torch.linspace(...).round()`로 선택한다. 3초마다 한 프레임을 뽑는 방식이 아니다.
- 기존 Qwen 최소 프레임 수/짝수 정렬을 유지한다. 보통 3초 clip은 **6프레임**, 짧은 tail은 **4프레임**이며 실제 effective fps는 clip별 기록에서 확인한다. 아주 짧은 segment는 실제 프레임 수에 맞춰 2프레임이 될 수 있다.
- 한 프레임뿐인 tail은 명시적으로 opt-in한 reader 경로에서 같은 프레임을 2회 사용해 temporal patch 조건을 충족한다. 인접 구간 프레임은 가져오지 않는다. 이번 5개에는 해당 tail이 없다.
- `min_pixels:4096`, `max_pixels:393216`, `total_pixels:33554432`, image patch size 16. 실제 해상도는 **832 × 448**.
- Processor `do_sample_frames=False`, `do_resize=False`; 계측한 바로 그 입력 tensor로 generate한다.
- 기존 생성 설정 유지: `do_sample:false`, `max_new_tokens:4096`, `repetition_penalty:1.0`.
- Prompt는 사용자 요청 원문을 `prompts/virat_clip_caption_pilot.txt`에 저장한다. 모델에는 segmentation 경계나 원본 video 길이를 prompt로 주지 않는다. Qwen 기본 video processor의 clip 내부 프레임 timestamp 표현은 그대로 사용한다.

## 출력

기본 폴더: `outputs/virat_clip_caption_pilot/`

- `captions/<원본 filename>.json`: `video`, `clip_duration_sec:3`, `clips:[{start_sec,end_sec,caption}]`. caption은 원문 문자열 그대로이며 여러 sentence/줄바꿈을 보존한다.
- `review/<원본 filename>.txt`: `[00:00–00:03]` 형식의 빠른 검토용 텍스트.
- `inference/<video stem>/<clip index>.json`: clip별 원문, source frame 범위, 실제 sampling frame indices/clip 내부 시각/fps/해상도/processor grid, 입력·출력 토큰 수, 소요 시간, 정상/잘림 상태.
- `segments/<video stem>/<clip index>.avi`: sampling 전 실제 video segment.
- `manifest.json`: prompt, model/generation/video 설정, source 파일 identity, 모든 segmentation 경계 및 fingerprint.
- `results.json`: 영상별 완료 clip 기록 목록, clip 상태별 개수와 전체 처리 완료 여부. `complete:true`는 51개 처리 완료를 뜻하며 모든 응답이 정상이라는 뜻은 아니다. 중단 시 영상별 caption JSON/TXT는 저장된 clip까지 포함하므로 완료 여부는 이 파일 및 inference 기록으로 확인한다.
- `dry_run/`, `dry_run.json`: 모델 생성 없이 실제 processor 입력 측정.
- `pilot.log`: 실제 실행 로그.
- `environment.json`: 실제 실행 패키지/CUDA 버전.
- `segment_verification.json`: 이번 실행의 전체 clip frame count 및 원본 대비 첫/마지막 decoded RGB pixel 비교 결과.

이 결과는 짧은 event recall, participant count, 연속 관찰의 일관성, hallucination을 사람이 비교하기 위한 관찰 자료다. clip이 짧아도 sampling 간격 사이의 event나 축소된 작은 객체를 놓칠 수 있다. 개선 여부를 자동으로 판정하지 않는다.

## 검증

```bash
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_clip_caption.py'
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_model_video_measurement.py'
```
