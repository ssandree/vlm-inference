# Text-only temporal event merge pilot

기존 clip caption JSON의 timestamp와 caption만 Qwen3-VL-30B-A3B-Instruct에 전달한다. 출력은 사람이 원본 영상과 비교하여 수정할 event candidate이며 GT가 아니다. video decoding/frame sampling/ROI/video inference는 실행하지 않는다.

## 실행

```bash
cd /home/ssandree/Motion-guided-ROI/vlm-inference
.vlm-inf/bin/python scripts/run_virat_temporal_event_merge.py

# 입력과 실제 prompt만 검증; 모델 로딩 없음
.vlm-inf/bin/python scripts/run_virat_temporal_event_merge.py --dry-run

# 재실행은 기존 결과를 보존하는 새로운 destination 사용
.vlm-inf/bin/python scripts/run_virat_temporal_event_merge.py \
  --output-dir outputs/virat_clip_caption_pilot/merged_events_rerun
```

기본 입력은 `outputs/virat_clip_caption_pilot/captions/VIRAT_S_010001_03_000537_000563.mp4.json` 하나다. `--input`은 단일 JSON 파일만 받으며 directory/batch 모드를 제공하지 않는다. merge config의 `jobs: []`를 강제한다. 이번 실행은 다른 네 영상에 적용하지 않는다.

## Prompt와 모델

- `prompts/virat_temporal_event_merge.txt`: 사용자 제공 12가지 규칙과 JSON 예시. 예시가 실제 관측 근거가 아니라는 점, source ID를 그대로 사용할 것, 관측이 없는 gap을 건너뛰지 말 것, 한 clip이 여러 동시 event를 지원할 수 있다는 점을 덧붙였다. 최초 응답이 여러 행동을 시간 block으로 묶어 반환하여, 각 JSON object를 하나의 subject/group + action event로 제한하고 상충하는 방향/인원 수를 강제로 통합하지 말라는 지시도 보강했다.
- 실제 input에는 각 caption 원문과 `source_clip`, `start_sec`, `end_sec`를 JSON으로 첨부한다. 마지막 구간 ID도 원본 경계의 전체 float 표현을 보존한다: `18-20.69253233208177`.
- 기존 `Qwen3VLCaptioner.generate_text` / `load_config` / `30b` preset을 재사용한다. loader는 이 CLI에서만 `allow_empty_jobs=True`로 호출하며 기존 video CLI의 jobs 필수 조건은 유지한다. BF16, device_map auto, SDPA, INT8/FP8 대체 없음.
- 새 merge config: `configs/virat_temporal_event_merge.yaml`. `do_sample:false`, `repetition_penalty:1.05`, `max_new_tokens:2048`. 구조화된 여러 event를 반환하는 text 단계의 독립 생성 설정이다.
- `generate_text`에 기존 video generate와 동일한 출력 토큰/EOS 한도 계측을 추가했다. visual token count는 0이어야 한다.

## 검증과 출력

`outputs/virat_clip_caption_pilot/merged_events/<원본 video filename>` 접두사로 저장한다.

- `.json`: `{"video": "...mp4", "events": [{"start_sec": ..., "end_sec": ..., "sentence": "...", "source_clips": [...]}]}`.
- `.md`: 원래 7개 clip caption, Qwen event 후보, 각 event의 source_clips, clip-to-event 표.
- `.prompt.txt`: 실제 전달한 전체 text prompt.
- `.raw.txt`: Qwen 원문 JSON 응답.
- `.validation.json`: JSON/schema/시간 경계/source_clips 검증 결과. rejected event는 원형과 오류를 보존한다.
- `.metadata.json`: 입력 파일 SHA256, model/generation 설정, text-only 표기, 입력/출력 토큰 수, loading/generation 시간, 잘림 여부.

기존 clip JSON과 이미 존재하는 merge 산출물은 덮어쓰지 않는다. raw 응답이 잘리거나 하나라도 잘못된 event가 있으면 검수용 JSON/MD를 발행하지 않고 원문과 검증 결과만 보존한다. 자동 재시도나 다른 모델로 대체하지 않는다.

각 event는 정확히 네 필드만 허용한다. source_clips는 실제 입력 ID이며 중복 없이 시간순으로 연속되어야 한다. start/end는 해당 clips의 최초/최종 경계와 같아야 한다. 부동소수점 표기 차이 1e-9 이내만 허용하고 저장은 원본 segmentation 경계를 사용한다. gap 병합, 임의의 시간 세분화, 알 수 없는 source ID를 거부한다. 서로 다른 event는 동일 clip을 공유할 수 있으며 시간 구간도 겹칠 수 있다.

이 검증은 provenance 구조와 시간 경계를 확인한다. 문장의 실제 사실성, 같은 사람인지 여부, 모든 관측의 보존 여부는 자동 증명하지 않는다. 원래 caption의 오류가 병합 결과에 이어질 수 있으므로 사람이 영상과 대조한다.

## 테스트

```bash
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_temporal_event_merge.py'
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_model_video_measurement.py'
```

## 이번 실행의 보존 자료

`merged_events/initial_attempt/`에 최초 Qwen 응답과 검증/metadata/비교 문서를 보존했다. 최초 결과는 구조적 provenance 검증을 통과했지만, 6–9초의 올라가는 행동을 내려가는 event에 포함하고 독립 행동을 같은 object에 묶었다. `review_notes.txt`에 이 문제를 기록했다. prompt를 보강한 재실행도 동일 영상 하나에만 적용한다.

최종 재실행에서는 7개 입력 clip에서 12개 event candidate를 생성했다. 입력 SHA256 일치, Qwen 원문 대비 event 보존, source_clips/시간 경계, visual token 0, 토큰 한도 미도달을 확인했다. 모델 로딩 322.90초, text generation 62.42초, 출력 827토큰이다. 여러 clip을 연결한 event는 우산 아래 앉아 있는 사람 하나이며, 같은 내려가기/올라가기 행동을 일부 분리해 둔 과소 병합이 남았다. `comparison_check.json`과 `.md` 검토 메모에 기록했다.

## Aggressive prompt 변경

현재 `virat_temporal_event_merge.txt`는 사용자 요청의 간결한 aggressive 규칙이다. 이전 보수적 원문은 `virat_temporal_event_merge_conservative.txt`에 보존했다. `build_prompt`는 입력 뒤에 독립 활동별로 event object를 분리하고 관측 누락으로 등장/소멸을 추론하지 말라는 짧은 지시를 덧붙인다. JSON code fence는 형식만 제거해 파싱하며 event 내용은 수정하지 않는다. 59개 전체 실행은 [pipeline 문서](virat_clip_caption_pipeline.md)를 참고한다.

현재 builder는 각 clip caption을 원문 문장별 observation record로 나누어 입력하며 action taxonomy나 heuristic merge를 적용하지 않는다. Aggressive pilot의 과도한 장면 요약/복합 event 및 요청한 연속 event 미병합은 `merged_events_aggressive_v4/pilot_evaluation.json`에 기록했다. 59개 실행의 조건을 충족하지 못하여 전체 inference를 시작하지 않았다.
