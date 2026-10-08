# VIRAT 30-second intermediate event consolidation

Reads only existing `outputs/virat_clip_caption/captions/<video>.json` files for the selected 59 videos. Never decodes videos/frames, generates 3-second captions, or writes to the source directory. All required caption JSONs must exist and pass the existing consecutive-3-second validation before model loading.

Windows are `[0,30)`, `[30,60)`, etc.; the final partial window is retained. Every source clip in each window appears in the prompt verbatim with its timestamps. The requested prompt is stored in `prompts/virat_intermediate_30s.txt` and formatted without adding instructions. The output JSON has `video`, `window_duration_sec`, source path/hash, and `windows`; each window stores `start_sec`, `end_sec`, `intermediate_caption`, all `source_clips`, and the complete `source_segments` evidence. Source evidence applies to the entire window, not sentence-level attribution.

Configuration: `configs/virat_intermediate_30s.yaml`, Qwen3-VL-30B-A3B-Instruct, original BF16 loading utility, text-only `generate_text`, greedy decoding, max_new_tokens=2048, repetition_penalty=1.2. A single model instance handles all windows and the pilot evidence check.

The first video is `VIRAT_S_010001_03_000537_000563.mp4`. Validate nonempty/nontruncated generation and unchanged input hashes, then use an additional text-only model call to check that distinct input events are preserved, unsupported claims are absent, and uncertainty/order are retained. Store the complete audit in `inference/<video>/0000.audit.json`. Only a passing pilot proceeds to the remaining videos. This is automated evidence verification; it does not establish agreement with the original video or produce GT. A failed pilot stops the run with artifacts preserved.

```bash
.vlm-inf/bin/python scripts/run_virat_intermediate_30s.py
```

Outputs: `outputs/virat_intermediate_30s/captions/<video>.json` and `.md`; prompt/attempt/record artifacts under `inference/`; `manifest.json`, `preflight.json`, `pilot_validation.json`, `progress.json`, `pipeline.log`. Normal window records automatically resume without inference; changed inputs/settings require a new output directory. Truncated/empty records are retained as numbered attempts and not published as successful window results.

Read-only preflight for an existing individual pilot:

```bash
.vlm-inf/bin/python scripts/run_virat_intermediate_30s.py --pilot-only --dry-run --output-dir outputs/virat_intermediate_30s_preflight
```

The local one-shot scheduler `scripts/schedule_virat_intermediate_30s.py --start-at <unix-seconds>` waits until the scheduled time, then acquires the source pipeline's existing lock. If the 145-clip retry is still running, it waits without competing for GPU resources. It requires retry stage `complete` and zero remaining failures, then launches the consolidation pipeline. Missing/invalid source files or failed retry never trigger regeneration. Scheduler states are stored in `schedule.json`; `scheduler.log`, `scheduler.pid`, `launcher.log`, and `run.pid` record actual execution. This is a detached local process, not a reboot-persistent system service.
