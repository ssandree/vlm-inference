# Independent VIRAT event generation: A vs B

Source: completed `outputs/virat_clip_caption/captions/<video>.json` files (59 videos). The script reads and hashes these existing files; it never writes to the source directory or loads video/frame inputs. Original 3-second observations, including their omissions and uncertainty, are preserved unchanged.

- A: fixed 30-second windows → new event-preserving descriptions → video-level event JSON. Intermediate generation uses the NEW A1 prompt from the request, not the earlier consolidation output. A2 receives only timestamps plus intermediate descriptions. Across-window events can merge; independent actions can split within a window.
- B: full original timestamped 3-second observations → event JSON. B receives no A output or intermediate descriptions and is a separate fresh text call with no conversation state. `--pipeline B` works without executing A; `--pipeline A` works without executing B.

Exact requested prompt templates: `prompts/virat_ab_a1_30s.txt`, `prompts/virat_ab_a2_events.txt`, and `prompts/virat_ab_b_direct.txt`. Literal JSON schema braces are preserved during substitution. No extra LLM judge or semantic quality ranking is called.

Final JSON in BOTH branches is exactly:

```json
{"events": [{"start": 0, "end": 30, "description": "An observable event."}]}
```

Source/provenance, inference settings, raw generations and validations live in separate files, not additional final event fields. A intermediates contain `window_start`, `window_end`, `description`, plus all `source_clips` and verbatim `source_segments`. Generated event output is for human verification; it is not verified GT.

Validation checks JSON parsing, exact common schema, finite positive intervals, chronological start ordering, exact duplicate event objects, missing/duplicate windows, complete source-clip coverage, unchanged input bytes, and output files. Simultaneous events may overlap. A event timestamps must use window endpoints or explicit times retained in the intermediate descriptions, intersected with actual source clip boundaries; B uses available 3-second boundaries. The model output is not silently rounded, rewritten or heuristically repaired. A cannot use fine timing that was lost in intermediate text; timestamps explicitly retained in A1 remain usable evidence. Semantic omissions/hallucinations are left for human comparison, not an LLM judge.

Pilot (ONE representative video, ~40.97s, two windows):

```bash
.vlm-inf/bin/python scripts/run_virat_event_ab.py
```

Default output: `outputs/virat_event_ab_pilot/`:

```text
A_hierarchical/
  intermediate/<video>.json
  events/<video>.json
  inference/<video>/A1/...
  inference/<video>/A2/...
B_direct/
  events/<video>.json
  inference/<video>/B/...
comparison/<video>.md
validation/<video>.json
manifest.json
progress.json
pipeline.log
```

Configuration: `configs/virat_event_ab.yaml`, Qwen3-VL-30B-A3B-Instruct via existing loading/generate_text utility, BF16, greedy decoding, repetition_penalty=1.2. A1 max_new_tokens=2048; A2/B max_new_tokens=max(4096,min(16384,1024+64*input_count)). Model is loaded once per invocation and reused, but each request is independently constructed. Failed attempts retain exact raw outputs/prompts. Resume accepts only matching source hashes, templates and settings. Existing results require `--resume` or a new output directory.

Full run commands below are provided for a later explicit instruction; pilot never starts a full run automatically:

```bash
.vlm-inf/bin/python scripts/run_virat_event_ab.py --all --output-dir outputs/virat_event_ab_full
```

This runs A and B with separate branch directories and also generates per-video comparison MDs. To execute each branch alone in entirely separate top-level directories:

```bash
.vlm-inf/bin/python scripts/run_virat_event_ab.py --all --pipeline A --output-dir outputs/virat_event_A_full
.vlm-inf/bin/python scripts/run_virat_event_ab.py --all --pipeline B --output-dir outputs/virat_event_B_full
```

Do not run two heavy model instances concurrently on the same GPU. A combined invocation avoids redundant loading while keeping all branch inputs and artifacts separate.
