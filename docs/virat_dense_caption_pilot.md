# VIRAT dense-caption GT candidate pilot

`Qwen3-VL-30B-A3B-Instruct`로 원본 영상 전체를 한 번에 입력하고 시간순 사건 후보를 생성한다. 최종 GT가 아니며 실제 사건 여부와 시간 경계는 사람이 검수한다. annotation/GT, 이전 캡션, ROI/track 결과는 모델에 전달하지 않는다.

## 실행

저장소 루트에서 실행한다. 기존 `.vlm-inf` 환경과 로컬 30B BF16 체크포인트를 사용한다.

```bash
cd /home/ssandree/Motion-guided-ROI/vlm-inference

# 모델 가중치 없이 실제 영상/processor 입력 검증
.vlm-inf/bin/python scripts/run_virat_dense_caption_pilot.py --dry-run

# 대표 5개 실제 추론
.vlm-inf/bin/python scripts/run_virat_dense_caption_pilot.py

# 완료된 동일 설정 결과는 유지하고, 남은 영상 실행
.vlm-inf/bin/python scripts/run_virat_dense_caption_pilot.py --resume

# prompt/설정 변경 후 재실행할 때는 새 출력 폴더 사용
.vlm-inf/bin/python scripts/run_virat_dense_caption_pilot.py \
  --output-dir outputs/virat_dense_caption_pilot_rerun
```

전처리와 추론 산출물은 별도 디렉터리이므로 같은 출력 폴더에서 dry-run 후 inference를 실행할 수 있다. 기존 파일이 있는 dry-run은 `--dry-run --resume`을 사용한다. `--limit 1`로 첫 영상만 별도 출력 폴더에서 먼저 실행할 수 있다. 명시적인 video job만 허용하고 최대 5개로 제한해 전체 dataset 실행을 방지한다. `--limit`을 바꾸면 선택 영상 목록과 fingerprint가 바뀌므로 새 출력 폴더를 사용한다.

`--resume`은 완료되고 설정이 일치하는 영상만 건너뛰며 검수 파일을 덮어쓰지 않는다. 실패/중단되어 metadata가 없는 일부 산출물이나 설정 불일치를 발견하면 새 출력 폴더를 요구한다. 여러 프로세스로 같은 출력 폴더에 동시에 실행하지 않는다.

## 대표 영상과 실제 입력

선택 이유는 pilot의 장면 다양성이다. 표의 장면 이름은 원하는 event를 모델에 암시하는 prompt로 쓰지 않는다.

| 원본 filename | 선택 이유 | 길이(s) | 실제 프레임 | 실제 fps | Qwen 전처리 해상도 |
|---|---|---:|---:|---:|---|
| VIRAT_S_010001_03_000537_000563.mp4 | 계단과 여러 보행자 | 20.692532 | 40 | 1.933065 | 832 × 448 |
| VIRAT_S_010106_05_000954_000996.mp4 | 주차장, 차량 주변/물체 행동 검토 | 27.409262 | 54 | 1.970137 | 832 × 448 |
| VIRAT_S_010113_05_000776_000805.mp4 | 원거리 이동수단 | 22.778473 | 44 | 1.931648 | 832 × 448 |
| VIRAT_S_010208_09_000857_000886.mp4 | 건물 입구와 보행 | 20.358782 | 40 | 1.964754 | 832 × 448 |
| VIRAT_S_040005_00_000003_000058.mp4 | 다른 카메라의 차량 이동 | 54.721333 | 108 | 1.973636 | 832 × 448 |

측정값의 기준 파일은 `outputs/virat_dense_caption_pilot/dry_run.json`과 실제 추론 `inference/*.metadata.json`이다. fps의 소수점 표시는 반올림이며, 정확한 값은 JSON에 저장한다.

- 원본 전체 길이는 디코더의 frame count / native fps로 계산한다. filename의 시간 숫자를 사용하지 않는다.
- `fps: 2.0`을 요청한다. 기존 PyAV reader가 Qwen `smart_nframes`로 프레임 수를 정하고, 전체 첫/마지막 프레임을 포함하는 `torch.linspace(...).round()` 인덱스로 균등 선택한다. 잘라낸 앞부분만 보거나 서로 독립적인 window caption을 합치는 방식이 아니다.
- 실제 fps는 `선택 프레임 수 × native fps / source frame count`이다. 프레임 수의 짝수 정렬 때문에 요청값과 다르다. 실제 인덱스·프레임 시각·평균 간격을 metadata에 기록한다.
- Qwen에 원본 fps와 실제 source frame indices를 전달해 영상 시작 기준 timestamp를 유지한다. processor의 `do_sample_frames=False`, `do_resize=False`로 이중 샘플링/리사이즈를 막는다. 실제 측정한 processor 텐서를 그대로 `generate`에 전달한다.
- `max_pixels: 393216`, `total_pixels: 33554432`, image patch size 16. Qwen utils의 aspect-ratio 유지 및 32픽셀 배수 정렬로 실제 입력은 위 해상도가 된다. `total_pixels`는 Qwen utils의 budget 파라미터이며 최종 전체 RGB tensor pixel 수와 같은 의미로 간주하지 않는다.
- `video_grid_thw`와 모델의 temporal patch 이후 padding 포함 frame count도 기록한다. 이 5개 영상은 추가 temporal padding이 없으며 각각 40/54/44/40/108프레임이다.
- `do_sample:false`, `max_new_tokens:4096`, `repetition_penalty:1.0`으로 실행한다. 토큰 한도 도달 여부와 실제 입력/출력 토큰 수를 기록한다.

영상 전체를 샘플링하지만 전 프레임을 모델에 전달하지는 않는다. 약 0.5초 사이에 끝나는 행동이나 축소 후 식별되지 않는 작은 객체는 누락될 수 있으므로, 이 결과를 그대로 최종 GT로 사용하지 않는다. 시간 경계는 선택된 프레임의 관측 근거에 따라 검수한다.

## 검수할 JSON

`outputs/virat_dense_caption_pilot/candidates/<원본 filename>.json`

```json
{
  "video_filename": "VIRAT_S_010001_03_000537_000563.mp4",
  "events": [
    {
      "timestamp": "12.4 - 16.8",
      "sentence": "A person walks from left to right across the upper part of the parking lot."
    }
  ]
}
```

위 내용은 형식 예시이며 실제 추론 결과가 아니다. 각 event에는 **timestamp와 sentence만** 있다. 사람이 두 값을 바로 편집하거나 event를 추가/삭제하면 된다. timestamp는 원본 clip 시작 기준 초 단위이며 동시 사건의 구간이 겹칠 수 있다.

별도 보존 자료:

- `results.json`: 완료 영상 목록, 파일 위치, 추론 상태 및 실제 입력 metadata.
- `inference/<filename>.raw.txt`: 모델 원문 응답.
- `inference/<filename>.prompt.txt`: 길이를 치환해 실제 전달한 prompt.
- `inference/<filename>.metadata.json`: 모델/생성 설정, 실제 프레임 인덱스·시각·fps·해상도·토큰 수.
- `inference/<filename>.validation.json`: 파싱 결과와 원형을 보존한 rejected events.
- `dry_run.json`, `dry_run/`: 모델 가중치를 로드하지 않은 실제 전처리 검증.
- `pilot.log`: 이번 실제 pilot 실행 로그.
- `environment.json`: 이번 실행 환경의 실제 PyTorch/Transformers/Qwen utils 등 패키지 버전.

JSON array 및 정확히 두 event 필드를 검증하고, `0 <= start < end <= duration`을 확인한다. 시간순 정렬을 제외하고 sentence 변경, 임의 timestamp clamp, 사건 합치기/중복 제거를 하지 않는다. 잘못된 event는 원문과 오류를 별도 보존한다. `partial` 결과는 유효 행만 candidates에 저장하고 종료 코드를 1로 반환한다. JSON/전체 schema 오류나 잘린 응답은 정상 candidate 파일로 발행하지 않는다. 문장 의미/사실성 및 '한 문장' 여부는 기계적 구두점 개수로 판정하지 않으며 사람이 검수한다.

## Prompt와 재사용 코드

- 설정: `configs/virat_dense_caption_pilot.yaml`
- Prompt 원문: `prompts/virat_dense_caption_pilot.txt`
- CLI: `scripts/run_virat_dense_caption_pilot.py`
- 두 필드 출력 검증: `src/vlm_inference/dense_caption.py`
- 모델 로딩/생성: 기존 `src/vlm_inference/model.py`의 `Qwen3VLCaptioner` 재사용. `prepare_multimodal_inputs`로 기존 processor 코드를 공유하고 실제 입력 계측을 추가했다.
- 영상 로딩: 기존 `src/vlm_inference/video_io.py`의 `inspect_video`와 PyAV Qwen backend 재사용.
- 모델 preset/config/job: 기존 `config.py`, `inputs.py` 재사용.
- 기존 복잡한 structured schema의 `dense_events.py`와 그 prompt/config는 변경하지 않았다.

Prompt는 실제 관찰 event만 기술하고 subject/action/target 및 보이는 공간·방향을 포함하도록 요청한다. conversation/intention/relationship 추론, 정적 배경 나열, 불필요한 걷기 분할을 금지한다. 서로 다른 시점·동시 독립 사건은 분리하며, 문/차량 진입과 물체 전달은 명시적 시각 근거가 있을 때만 허용한다. 먼 객체도 식별 가능한 경우 포함하고 hallucination 억제를 개수보다 우선한다. 전체 내용과 실제 길이를 치환한 버전은 위 prompt 파일들에서 확인한다.

Qwen의 metadata 전달과 재샘플링 비활성화는 [공식 Qwen3-VL 사용 예시](https://github.com/QwenLM/Qwen3-VL#quickstart)에 맞췄으며, 설치된 `qwen-vl-utils`/Transformers 코드를 확인해 계측했다.

## 검증

```bash
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_dense_caption.py'
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_model_video_measurement.py'
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_model_catalog.py'
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_dense_caption_pilot_cli.py'
```
