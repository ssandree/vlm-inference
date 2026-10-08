# Selected-59 VIRAT clip caption and temporal merge pipeline

`configs/virat_multi_roi_59.txt`의 기존 선정 59개만 사용한다. 총 1,100개의 연속 3초 구간이며 마지막 짧은 구간을 포함한다. 원본 길이는 native frame count/fps 기준이다.

## 실행과 재개

```bash
cd /home/ssandree/Motion-guided-ROI/vlm-inference
# 파일/길이/segmentation/설정 검증; 모델 추론 없음
.vlm-inf/bin/python scripts/run_virat_clip_caption_pipeline.py --dry-run
# 새 run 및 중단 후 정상 결과를 재사용하는 재개
.vlm-inf/bin/python scripts/run_virat_clip_caption_pipeline.py --resume
```

`--resume`은 동일 model/prompt/generation/video/source identity인 정상 clip 기록과 정상 merge 기록만 재사용한다. 설정이나 source가 달라지면 새 `--output-dir`를 요구한다. 잘림/빈 응답을 정상 완료로 취급하지 않는다. 비정상 clip은 다음 resume에서 재추론하고 기존 원문은 `.failed_N.json`에 보존한다. 정상 caption/merge JSON이 사람이 편집하여 기록과 다르면 자동으로 덮어쓰지 않고 중단한다. 누락된 export는 정상 기록으로 복구할 수 있다. 같은 output directory의 동시 실행은 파일 lock으로 차단한다.

## 순서와 모델 재사용

1. 각 source를 한 번 decode하여 clip별 FFV1 lossless AVI를 생성한다. 전체 689초짜리 video를 230번 다시 decode하는 비용을 피한다.
2. 각 clip을 독립 video input으로 caption한다. 이전 clip 응답은 다음 입력에 전달하지 않는다.
3. 정상 응답이 모든 clip에 있는 video별 caption JSON/TXT를 저장한다. 실행 중에는 clip metadata와 partial captions를 먼저 저장한다.
4. **59개 모든 caption이 정상 완료된 뒤** text-only temporal merge를 시작한다. 비정상 caption이 있으면 나머지 caption 작업은 진행하고 merge는 시작하지 않는다.
5. video별 merge JSON/MD와 source_clips, 원문·검증·metadata를 저장한다.

프로세스 전체에서 Qwen30B를 한 번만 로딩하고 processor/model을 모든 video 및 두 stage에서 재사용한다. caption이 모두 skip되어 merge만 필요한 resume은 merge stage에서 한 번 로딩한다. 전부 정상 완료된 resume은 모델을 로딩하지 않는다.

## 모델과 생성 설정

- 기존 `30b` preset: Qwen3-VL-30B-A3B-Instruct, BF16, SDPA, device_map auto.
- Caption: `configs/virat_clip_caption_59.yaml`; 기존 caption prompt 유지. 요청 fps 2.0, min/max/total pixels 4096/393216/33554432, do_sample false, repetition_penalty 1.05, max_new_tokens 512.
- Sampling: 기존 PyAV Qwen smart_nframes + linspace. 실제 입력 frames/fps/resolution/indices/tokens는 clip별 기록에 저장한다.
- Merge: `configs/virat_temporal_event_merge.yaml`, 사용자 제공 간결한 aggressive prompt. wording/location/direction/visible count의 작은 변화로 분할하지 않으며 자연스러운 동일 활동 흐름을 우선한다. 입력 뒤에 활동별 JSON object 분리와 관측 누락으로 등장/소멸을 추론하지 말라는 짧은 지시를 보강한다. 앉기와 걷기처럼 독립된 활동은 분리한 뒤 각각의 연속 활동을 적극적으로 병합한다.
- 입력은 원래 caption을 문장별 관측 record로 나누되 원문 문장과 원본 clip ID/경계를 보존한다. 독립 활동이 하나의 clip 문단으로 입력되어 전체 장면 요약으로 합쳐지는 현상을 줄이기 위한 입력 포맷이며, action 분류나 heuristic merge는 하지 않는다.
- Merge는 images/videos 없이 `generate_text`를 호출한다. 모델은 event를 추가하지 않고 기존 captions를 재정리한다. 최종 GT가 아닌 human-review candidate다.
- 긴 video의 source_clips 목록을 끝까지 기록할 수 있도록 merge 출력 한도는 video별 `max(4096, min(16384, 1024 + 48 * num_clips))`이다. 실제 출력이 한도에 도달하면 정상 결과로 발행하지 않는다.

보통 프레임은 native timestamp가 clip의 half-open 구간에 속하도록 중복 없이 나눈다. `VIRAT_S_010110_00_000000_000021.mp4`와 `VIRAT_S_010207_10_001549_001596.mp4`의 마지막 0.03초 미만 구간은 마지막 프레임의 표시 시간만 남는다. 이때 마지막 실제 프레임을 tail input에 재사용하며 `boundary_frame_reused:true`로 기록한다. 한 프레임 video는 Qwen temporal patch를 위해 동일 프레임을 2회 sampling하며 인접 구간의 새 관측을 만들지 않는다.

## 산출물과 진행 상태

기본 output: `outputs/virat_clip_caption/`.

- `captions/<video>.json`: `video`, `clip_duration_sec`, `clips:[{start_sec,end_sec,caption}]`; 정상 완료 video만 발행.
- `captions/<video>.txt`: 원래 caption 검토 자료.
- `merged_events/<video>.json`: `video`, `events:[{start_sec,end_sec,sentence,source_clips}]`.
- `merged_events/<video>.md`: 원래 caption, merged events, source_clips 및 clip-to-event 표.
- `inference/<video stem>/<clip index>.json`: clip 원문·상태·정확한 segmentation·실제 sampling/generation 기록.
- `partial_captions/<video>.json`: 진행 중·비정상 응답도 포함하는 중간 자료.
- `segments/<video stem>/<clip index>.avi`: 실제 lossless video input.
- `merge_inference/<video>.json`: 정상/비정상 merge 기록과 caption hash.
- `merge_attempts/<video>/<attempt>/`: 각 시도의 prompt, raw response, 검증/토큰/latency metadata. 항상 보존.
- `manifest.json`, `preflight.json`: 설정/원본 identity 및 59개/1,100개 확인.
- `progress.json`: 현재 stage/video/clip, 정상 caption clip/video 수, 실패 clip, merge 완료 수.
- `pipeline.log`: START/DONE 및 segmentation/model loading/stage barrier 기록.

JSON fence는 표현 형식만 벗겨 파싱할 수 있다. 문장 수정, heuristic event merge, 새로운 action 생성은 하지 않는다. source ID, 연속성, earliest/latest 경계를 검증하며 의미상 인물 동일성과 사실성은 사람이 확인한다.

## 검증

```bash
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_clip_pipeline.py'
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_clip_caption.py'
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_temporal_event_merge.py'
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_model_video_measurement.py'
```

pipeline integration test는 59개 작은 영상과 mock model로 stage barrier, 모델 1회 로딩, 추론 없는 정상 resume, 비정상 clip만 재추론, export 복구를 검증한다. 실제 PyAV test는 lossless 픽셀과 초단기 tail을 확인한다.

## 현재 pilot 판정과 전체 실행 상태

Aggressive prompt와 문장 단위 입력으로 기존 12개 후보가 5개로 줄었으나, E4+E5만 연결되고 E7+E8의 관측 보존 및 E10+E11+E12의 완전한 연결은 충족하지 못했다. 앉기/걷기 복합 event도 남았다. 따라서 사용자 요청의 정상 pilot 조건을 통과하지 못해 59개 inference는 시작하지 않았다. `pilot_gate.json`의 accepted_for_59_run=false를 발견하면 CLI도 model loading/inference를 시작하지 않는다. `progress.json`은 not_started_pilot_quality_failed를 기록한다. 최종 비교 자료는 `outputs/virat_clip_caption_pilot/merged_events_aggressive_v4/`에 보존되어 있다.
