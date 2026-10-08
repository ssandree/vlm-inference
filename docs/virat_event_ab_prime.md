# A′/B′ semantic-event pilot

This experiment preserves existing 3-second captions, earlier A/B outputs, and the earlier 30-second prose results. Only `VIRAT_S_010113_02_000434_000479.mp4` is supported by the pilot CLI; there is no full-dataset flag.

Read-only source: `outputs/virat_clip_caption/captions/VIRAT_S_010113_02_000434_000479.mp4.json` (14 clips, 40.96787651230705 seconds). Fixed local windows: 0–30 and 30–40.96787651230705. Each original caption is inserted verbatim with its absolute timestamps.

Exact supplied prompts, unchanged:

- `prompts/virat_abc_a_3s_to_30s_json.txt`: local Event JSON, no prose intermediate.
- `prompts/virat_abc_b_30s_to_full_json.txt`: all chronological local event JSONs plus window timestamps; no original captions or earlier prose output.
- `prompts/virat_abc_c_3s_to_full_json.txt`: all original 3-second observations only; no A′ output.

A′: independent local extraction per window → cross-window reconciliation → final events. B′: independent full-timeline extraction. Each text call starts a fresh user conversation. A single Qwen3-VL-30B-A3B-Instruct model instance is reused, using the repository's BF16 loading and `generate_text` utilities. No video/frame decoding, caption regeneration, heuristic event merging, or LLM quality judge is performed.

```bash
.vlm-inf/bin/python scripts/run_virat_event_ab_prime.py
```

New output root: `outputs/virat_event_ab_prime_pilot/`:

```text
A_hierarchical/
  local_30s/<video>/0000.json
  local_30s/<video>/0000.metadata.json
  local_30s/<video>/0001.json
  local_30s/<video>/0001.metadata.json
  final/<video>.json
  inference/<video>/A1/...
  inference/<video>/A2/...
B_direct/
  final/<video>.json
  inference/<video>/B/...
comparison/
  <video>.md
  <video>.statistics.json
manifest.json
validation.json
execution.json
progress.json
pipeline.log
```

Every local event export and both final exports use exactly `{"events": [{"start": <seconds>, "end": <seconds>, "description": "..."}]}`. Local-window boundaries and source evidence are in sidecars. Raw generations/prompts and per-attempt metadata are retained separately.

Validation uses ORIGINAL 3-second boundaries for all stages, not only 30-second boundaries. Final timestamps must be within the source duration; local timestamps must also lie within the local window. `start <= end` is allowed, overlapping events are allowed. Exact duplicate event objects are REPORT-ONLY and are retained; chronological-order warnings are also report-only. No timestamp rounding, sorting, duplicate deletion or event-description rewriting occurs. Source bytes are compared before completion. Existing exports cannot be overwritten with different values.

The comparison MD contains original 3s captions, each A′ local JSON, A′ final JSON, B′ final JSON, and a mechanical-statistics table. Statistics include source segments, all local events, both final event counts, events lasting strictly >3s/>6s, positive-duration overlap pairs (touching endpoints do not count), and conservative lexical static/no-action flags. The static count is a text-pattern diagnostic, not semantic verification; matching indices and method are retained in statistics JSON. No semantic winner is chosen.

Configuration: `configs/virat_event_ab_prime.yaml`, greedy decoding, repetition_penalty=1.2, local max_new_tokens=4096; final budget=max(4096,min(16384,1024+64*input_event_or_clip_count)). Prompts/source hashes and settings are captured in manifest and inference records. Normal generations resume without inference:

```bash
.vlm-inf/bin/python scripts/run_virat_event_ab_prime.py --resume
```

Dry-run prompt preparation (no model inference):

```bash
.vlm-inf/bin/python scripts/run_virat_event_ab_prime.py --dry-run --output-dir outputs/virat_event_ab_prime_preflight
```

The dry-run A2 prompt has an empty input placeholder because no local inference was run. The actual A2 inference receives generated local event JSONs.
