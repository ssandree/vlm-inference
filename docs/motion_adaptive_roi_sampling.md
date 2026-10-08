# Motion-adaptive ROI sampling

이 문서는 실험 기록이다. 실행 스크립트, 설정, `outputs/` 아래 crop·캐시·캡션은 지웠다.

ROI tube의 위치와 Global 프레임은 고정하고, ROI crop을 뽑는 시각만 모션에 따라 바꾼 실험이다. 모델은 Qwen3-VL-8B-Instruct-BF16, 프롬프트는 `prompts/frames_roi.txt`, 대상은 VIRAT 9개 영상과 Stage3 ROI 11개다.

모션 점수는 Stage3 튜브 구간에서 magnitude가 있는 프레임의 셀 magnitude 평균이다. 11개 ROI 점수의 33·66 백분위로 low / mid / high를 나눈다.

| 영상 | ROI | 분위 | 모션 점수 |
| --- | ---: | --- | ---: |
| VIRAT_S_000201_02_000590_000623 | 1 | high | 6.36 |
| VIRAT_S_000200_01_000226_000268 | 1 | high | 6.53 |
| VIRAT_S_000201_06_001354_001397 | 1 | mid | 3.83 |
| VIRAT_S_010003_08_000739_000778 | 1 | low | 1.96 |
| VIRAT_S_010004_00_000064_000100 | 1 | mid | 3.63 |
| VIRAT_S_010106_05_000954_000996 | 1 | mid | 2.54 |
| VIRAT_S_010106_05_000954_000996 | 2 | low | 0.49 |
| VIRAT_S_010200_05_000658_000700 | 1 | high | 3.92 |
| VIRAT_S_040000_02_000111_000145 | 1 | low | 1.87 |
| VIRAT_S_040000_02_000111_000145 | 2 | low | 0.38 |
| VIRAT_S_050000_05_000696_000732 | 1 | high | 17.46 |

캡션 54건은 모두 생성됐다. 자동 캡션 점수는 저장하지 않았다.

## 1. Global 0.25 fps + ROI 샘플링 비교

설정: `configs/feasibility_motion_adaptive.yaml`. Global은 데이터셋 0.25 fps, 입력 순서는 `global_anchor`. ROI geometry는 바꾸지 않았다. 추론은 2026-09-06(다섯 조건)과 2026-09-07(2 fps 상한)이다.

| 조건 | 규칙 | ROI 프레임 | Visual token |
| --- | --- | ---: | ---: |
| 고정 1 fps | 튜브 구간을 1초 간격 | 349 | 87,724 |
| 누적 τ=8, 최소 간격 0.5초 | magnitude를 더해 8 이상이면 선택. 선택은 0.5초 이상 떨어뜨림 | 430 | 110,880 |
| 적응 0.5 / 1 / 2 | low, mid, high에 0.5, 1, 2 fps | 441 | 114,296 |
| 고정 2 fps | 튜브 구간을 0.5초 간격 | 681 | 151,616 |
| 누적 τ=8 | 위 누적 규칙. 간격 상한 없음. 첫·마지막 프레임 포함 | 697 | 171,874 |
| 적응 0.5 / 1 / 4 | high만 4 fps | 724 | 180,462 |

Global 이미지는 모든 조건에서 82장이다. 0.5초 상한을 둔 누적 샘플링은 고정 2 fps보다 ROI 프레임이 37%, visual token이 27% 적다. 상한이 없으면 `050000` (점수 17.46)이 178장, 43,920 token으로 고정 2 fps(72장)를 넘긴다.

영상별 ROI 프레임:

| 영상 | 고정 1 | 상한 | 0.5/1/2 | 고정 2 | 누적 | 0.5/1/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 000623 | 34 | 54 | 66 | 66 | 94 | 131 |
| 000268 | 43 | 66 | 85 | 85 | 122 | 168 |
| 001397 | 44 | 62 | 44 | 86 | 79 | 44 |
| 000778 | 39 | 40 | 20 | 76 | 40 | 20 |
| 000100 | 28 | 42 | 28 | 55 | 49 | 28 |
| 000996 | 53 | 31 | 40 | 103 | 38 | 40 |
| 000700 | 34 | 41 | 66 | 66 | 61 | 130 |
| 000145 | 37 | 34 | 20 | 72 | 36 | 20 |
| 000732 | 37 | 60 | 72 | 72 | 178 | 143 |

캡션 평균 길이: 고정 1 fps 519자, 적응 0.5/1/2 531자, 고정 2 fps 587자, 적응 0.5/1/4 589자, 2 fps 상한 627자, 상한 없는 누적 707자. 2 fps 상한만 지연 시간이 남아 있고 영상당 평균 19.8초다.

같은 계열로 준비한 조건:

- `burst_2fps`: 배경 1 fps, magnitude가 튜브 75 백분위를 넘는 구간이 2초 이하일 때만 그 구간을 2 fps. 긴 고모션 구간은 1 fps에 남긴다. 설정 `configs/feasibility_burst_2fps.yaml`.
- piecewise fps: 모션이 아니라 영상 시각 구간을 직접 지정해 fps를 바꾼다. 설정 `configs/feasibility_roi_piecewise.yaml`.
- τ sweep: 누적 규칙의 τ를 4, 6, 8, 10, 12로 두고 Global은 ROI 시작의 1 fps 풀에서 고른 `start_only` 한 장. 스크립트 `scripts/run_tau_sweep.py`. 이 기록 시점의 `outputs/tau_sweep/`은 이미 없었다.

## 2. Global은 영상 시작 1장, ROI만 네 조건

설정: `configs/feasibility_roi_sampling.yaml`. 스크립트 `scripts/run_roi_sampling.py`. Global은 영상 시작에 가장 가까운 1 fps 프레임 1장. ROI 조건은 고정 1 fps, 고정 2 fps, 누적 τ=8, centroid 2 fps다. centroid 2 fps는 기본 1 fps이고, ROI 중심 이동이 큰 주변만 2 fps다.

9개 영상 ROI 프레임 합: 고정 1 fps 349, centroid 2 fps 588, 고정 2 fps 681, 누적 τ=8 697.

| 영상 | 고정 1 | centroid 2 fps | 고정 2 | 누적 |
| --- | ---: | ---: | ---: | ---: |
| 000623 | 34 | 67 | 66 | 94 |
| 000268 | 43 | 72 | 85 | 122 |
| 001397 | 44 | 80 | 86 | 79 |
| 000778 | 39 | 78 | 76 | 40 |
| 000100 | 28 | 51 | 55 | 49 |
| 000996 | 53 | 84 | 103 | 38 |
| 000700 | 34 | 55 | 66 | 61 |
| 000145 | 37 | 48 | 72 | 36 |
| 000732 | 37 | 53 | 72 | 178 |

## 3. Spatial occupancy로 1 fps / 2 fps

optical flow magnitude가 아니라, 고정 ROI bbox 안 active block 분포가 바뀌면 2 fps로 올린다. 기본은 1 fps.

점수 `S = 0.4*Dc + 0.5*Dp + 0.1*Dr`. `Dc`는 0.6초 전후 centroid 이동, `Dp`는 3×3 histogram L1, `Dr`는 active ratio 변화다. 직전 3프레임 평균을 쓰고 padding은 없다. `S`가 `tau_high` 이상이면 2 fps, `tau_low` 이하이면 1 fps, 그 사이는 상태를 유지한다.

- 최초 명세: `tau_high=0.25`, `tau_low=0.15`. 9개 영상에서 high-fps 구간은 ROI 시간 331.7초 중 3.49초, 약 1.05%였다.
- 완화 설정: `configs/feasibility_roi_spatial_occupancy.yaml`의 `tau_high=0.17`, `tau_low=0.10`, `delta_sec=0.6`, `n_min=3`, `max_missing_frames=1`. Global은 `start_only`.

비교 파일은 `outputs/roi_spatial_occupancy/`와 `outputs/roi_spatial_occupancy_relaxed9/`에 있었다.
