# Spatial–Temporal Budget Trade-off 입력 계획 (1-5)

추론은 실행하지 않았다. 이 문서는 Clip 4개 × Budget 3개 × FPS 4개, 총 48개 입력 조건의 CPU 계산 결과다. 1-3과 1-4의 코드와 결과 파일은 수정하지 않았다.

## 질문

같은 visual-token 상한 안에서 FPS를 올리면 해상도를 얼마나 낮춰야 하는지, 그리고 그 상한이 실제로 채워지는지를 미리 고정한다. FPS와 해상도는 함께 정해지므로, 이후 인식 차이가 나도 해상도만 또는 FPS만의 효과로 단정하지 않는다.

## 데이터와 샘플링

클립은 1-4 설정과 같다. 샘플 시각은 `temporal_ablation.sampling.uniform_times`다. `t = clip_start + k / fps`이고 `t < clip_end`이다. 소스 프레임은 디코더 PTS가 가장 가까운 프레임이다. 이 규칙은 `select_frames`와 같고, 각 영상의 0.5 FPS 결과를 `select_frames`와 대조해 인덱스와 시각이 일치했다.

| Clip | 파일 | 구간 | 원본 | 0.5 / 1 / 2 / 4 FPS 프레임 |
|---|---|---|---|---|
| A | `VIRAT_S_000201_04_000682_000822` | 21–44s | 1280×720, 30 fps | 12 / 23 / 46 / 92 |
| C | `VIRAT_S_010106_05_000954_000996` | 2–22s | 1280×720 | 10 / 20 / 40 / 80 |
| D | `VIRAT_S_040005_00_000003_000058` | 14–31s | 1920×1080 | 9 / 17 / 34 / 68 |
| E | `VIRAT_S_010116_00_000061_000121` | 7–32s | 1280×720 | 13 / 25 / 50 / 100 |

프레임 수는 1-4의 `expected_frame_counts`와 같다. 소스 프레임 중복은 48개 조건 모두 없다. 목표 시각과 소스 PTS의 최대 절대 오차는 0.020755초다. 시각 전체는 `budget_feasible_configs.csv`의 `target_timestamps_sec`, `source_timestamps_sec`에 있다.

D는 1-4에서 1280×720으로 줄인 뒤 모델을 돌렸지만, 이번 상한은 원본 1920×1080이다. 업스케일은 하지 않는다.

## 해상도 탐색

콘텐츠 크기는 원본 가로세로비를 반올림으로 유지하고, 한 변은 32 이상, 원본을 넘지 않는다. 그 다음 1-4와 같이 32의 배수로 올려 padding한다. 720은 736이 되고, 1080은 1088이 된다. 이 padding 때문에 effective 가로세로비는 원본과 달라질 수 있다.

Visual token은 `ceil(frames / 2) * (H / 32) * (W / 32)`이다. 48개 모두 Qwen3-VL processor에 같은 크기의 영 텐서를 넣어 `video_grid_thw`와 placeholder 수가 이 식과 같은지 확인했다. `do_resize=False`, `do_sample_frames=False`이다. 언어 모델은 로드하지 않았다.

Budget 이하인 effective 면적 중 가장 큰 것을 고른다. 면적이 같으면 padding이 적은 콘텐츠, 그다음 토큰을 더 쓰는 쪽을 고른다.

## 실행 가능 여부

48개 모두 실행 가능하다. 최소 32×32 grid도 budget을 넘지 않아 실행 불가능 조건은 없다.

원본을 32 padding한 크기가 budget 안에 들어가는 조건은 15개다. 나머지 33개는 원본보다 작은 해상도로 내려야 한다. 원본이 들어가면 남는 토큰이 크다. C, 0.5 FPS, budget 20,480은 4,600토큰만 쓰고 15,880이 남는다. 이 칸은 같은 budget이라도 공간을 더 쓴 것이 아니다. 비교할 때 실제 토큰과 남은 토큰을 같이 봐야 한다.

Budget 20,480에서 원본 padding 크기가 들어가는 경우는 A 0.5·1 FPS, C 0.5·1·2 FPS, D 0.5·1 FPS, E 0.5·1 FPS이다. A 2 FPS는 1280×736이 21,160토큰이라 1280×704, 20,240토큰으로 한 단계 낮다. 4 FPS는 네 클립 모두 원본이 20,480을 넘는다.

## 메모리

추정은 끝난 1-4 run 24개의 peak allocated MiB를 visual token에 선형으로 맞춘 것이다. 식은 `16749.8776 + 0.316267 * visual_tokens` MiB이고, 기준 카드는 A6000 49,140 MiB이다. 1-4는 1280×736, bf16, sdpa, `max_new_tokens=256`이다. 토큰 수가 같으면 메모리도 같다는 가정이 들어가 있으며, 해상도 배치가 달라질 때의 차이는 새로 재지 않았다.

48개 추정의 범위는 18,188.9–23,201.7 MiB다. 전용 A6000에서는 모두 `low`다. `likely_oom`은 없다. 이 추정은 1-4의 짧은 생성 길이를 따른다. `max_new_tokens`를 2048로 늘리면 KV cache가 커지므로 이 숫자를 그대로 쓰면 안 된다.

계산 당시 GPU 1과 2는 다른 작업이 점유하고 있었다. 이번 스크립트는 `CUDA_VISIBLE_DEVICES=""`로 CPU만 사용했다. 나중에 추론한다면 비어 있는 GPU를 지정해야 한다.

## 이후 실행을 결정하기 전에

1-3과 1-4 결과를 본 뒤에 정할 항목은 다음이다.

- 48개를 모두 돌릴지, budget이 실제로 채워지는 칸만 돌릴지. 원본에서 이미 멈추는 칸은 FPS만 다르고 해상도는 같다.
- D를 원본 1920×1080 기준으로 둘지, 1-4처럼 1280×720 작업 해상도로 다시 제한할지.
- 생성 길이를 1-4의 256으로 유지할지. 메모 추정은 256 기준이다.
- 프롬프트는 1-4와 같은 문장을 yaml에 넣었다. 사건 수 제한이나 요약형으로 바꾸지 않았다.

추론 명령은 아직 없다.

## 산출물

- `experiments/qwen3vl_budget_tradeoff/budget_feasible_configs.csv`
- `experiments/qwen3vl_budget_tradeoff/budget_fps_resolution_heatmap.png`
- `experiments/qwen3vl_budget_tradeoff/budget_experiment_config.yaml`
- `experiments/qwen3vl_budget_tradeoff/scripts/prepare_budget_configs.py`

같은 계산을 다시 하려면 GPU 없이 다음을 실행한다.

```bash
cd /home/vailab02/Motion-guided-ROI/vlm-inference/experiments/qwen3vl_budget_tradeoff
CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false \
  /home/vailab02/anaconda3/envs/vlm_experiment_env/bin/python -u scripts/prepare_budget_configs.py
```
