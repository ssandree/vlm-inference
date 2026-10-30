# Gold lifetime-bbox crop captioning

일곱 VIRAT 영상을 lifetime bbox로 자르고, Qwen3-VL-8B-Instruct에 1, 2, 5, 8 fps로 넣어 영어 캡션을 만든 실험이다. 자동 ROI 검출은 하지 않는다. Full frame은 넣지 않는다.

## 실행

작업 디렉터리는 저장소 루트다.

```text
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
  /home/vailab02/anaconda3/envs/vlm_experiment_env/bin/python scripts/run_gold_lifetime_crop.py prepare

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
  /home/vailab02/anaconda3/envs/vlm_experiment_env/bin/python scripts/run_gold_lifetime_crop.py infer --fps 1,2,5,8
```

`status=ok` 체크포인트는 건너뛴다. 실패만 다시 돌리려면 `--retry-failed`를 붙인다. 성공분까지 덮어쓰려면 `--force`를 붙인다.

## 입력

- 모델: `Qwen/Qwen3-VL-8B-Instruct` revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- 디코딩: greedy, `max_new_tokens=768`, seed 0
- Processor: `do_sample_frames=false`, `do_resize=false`
- B: crop 두 개를 한 번의 generate에 넣는다. bbox를 합치지 않는다.
- A의 VLM content는 608×247, G는 1120×368이다. 같은 영상의 네 FPS가 그 크기를 공유한다. 이유는 `comparison.md`에 있다.
- 추론 픽셀은 원본 디코드 후의 in-memory crop이다. `crops/`의 mp4는 보관용이다.

## 산출물

| 경로 | 내용 |
|---|---|
| `captions/` | 조건별 캡션 원문 28개 |
| `results.csv`, `results.json` | 프레임 수, 토큰, 시간, 캡션 |
| `checkpoints/` | 재시작용 조건 기록 |
| `sampling/` | 목표 시각과 원본 프레임 인덱스 |
| `crops/`, `previews/` | lifetime crop 8개와 첫·중간 프레임 |
| `comparison.md` | 비교 보고서 |
