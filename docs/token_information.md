# Full-frame visual block information

Run one video, 5–10 event-relevant frames, layers 0/4/8:

```bash
.vlm-inf/bin/python scripts/analyze_token_information.py \
  --config configs/token_information_feasibility.yaml
```

Run one boundary, or change intervention batch size:

```bash
.vlm-inf/bin/python scripts/analyze_token_information.py \
  --config configs/token_information_feasibility.yaml \
  --layers 4 --batch-size 8 --output-dir outputs/token_information_layer4
```

The script reuses `Qwen3VLCaptioner`, its message builder, processor and model loader. Existing generation code is unchanged. Each frame is an independent Global image conversation, not ROI crops or temporally merged video tokens. The existing `prompts/frames_roi.txt` is used verbatim, including its mention of ROI images. No caption generation is needed: this analysis scores the ground-truth sentences marked visible at that timestamp.

## Configuration and GT

`token_information.frame_event_map` is the source of frames and matching. Edit `configs/token_information_frame_event_map.yaml` rather than guessing. Only `matched_event_ids` are scored. `uncertain_event_ids` are recorded in the manifest and ignored until you move them. Use existing `global_frame_..._t...jpg` names so the shared message builder emits Global timestamps. `frame_idx` in outputs is the zero-based selected-frame index, not the original video frame number. Feasibility is one video and 5–10 frames.

Each selected frame is teacher-forced on **only** the event sentences visible at that time, joined with one space. Scoring a still against the whole video-level paragraph is disabled. The matched sentences are tokenized once. The offset map assigns every caption token to exactly one sentence, including leading-space tokens. No EOS, assistant delimiters, or prompt tokens enter the GT score.

Each sentence gets its mean conditional token log probability. Each token is conditioned on all preceding caption tokens on that frame; sentences do not get independent forwards. Event-mean importance is kept as an auxiliary map. The primary multi-event map is `I_max = max` over matched events at that frame.

## Intervention

For each requested zero-based layer `l`:

1. One normal multimodal teacher-forcing pass captures entry to layer `l+1`, after layer `l` and its DeepStack addition. All requested boundaries are captured during this same pass. Caches live in CPU memory.
2. Physically remove all visual positions except the four belonging to a selected spatial block. Preserve all text positions, original rotary embeddings, and causal sequence order.
3. Run only decoder layers `l+1...L-1`, final norm, and LM head. The LM head operates only on positions predicting GT tokens.
4. Run the zero-visual-position suffix once per boundary for the baseline. It must be computed separately for each layer because cached text states differ.
5. Subtract the baseline sentence scores. Batch interventions of equal sequence length; vision and decoder prefixes are never rerun per block.

Qwen3-VL DeepStack adds features after its initial decoder layers. When any such additions remain after an intervention (notably `l=0`), only retained visual features are added. Removed visual tokens cannot reappear through DeepStack.

This implements the target-only versus text-only intervention of [Wang et al., Section 3.2](https://arxiv.org/html/2512.07580v2), extended here to four-token groups and sentence-mean log probabilities as requested. It is not leave-one-out attribution. It is also not an additive decomposition of the caption likelihood.

**Interpretation:** cached text states and retained visual states have already interacted with the entire full-frame input through layer `l`. The baseline therefore means *no remaining visual tokens at this boundary*, not a language-only model that never saw an image. Importance measures residual usefulness at that boundary. These are feasibility heatmaps of whether the motion ROI covers VLM-important 64×64 blocks, not validation of complete video-event coverage.

## Spatial mapping

Use `image_grid_thw / spatial_merge_size` to recover the actual merged token lattice. Group spatial neighbors `(2r,2c), (2r,2c+1), (2r+1,2c), (2r+1,2c+1)`; four consecutive flattened tokens are generally not a 2×2 block.

Each merged token is assigned exactly 32×32 pixels and each group 64×64 pixels, with no inverse resize mapping. The example increases the image pixel budget to retain full-frame detail; Qwen's ordinary dimension alignment still applies. Both original and mapped sizes are saved in `manifest.json`. Odd final token rows/columns cannot make a four-token block: those tokens are explicitly counted as ungrouped, excluded from the block domain, and removed during target-only interventions. `roi_token_ratio` uses **all** visual tokens as denominator; block coverage uses complete groups only.

`token_indices` are absolute positions in the full teacher-forced decoder sequence. `visual_token_indices` are indices into the concatenated visual-token array across frames.

## ROI input

Existing track bboxes, optionally selecting tube IDs:

```yaml
roi:
  tracks_file: /path/to/VIDEO_roi_tracks.json
  tube_ids: [1]
```

This reuses the repository's lifetime tube bbox loader. It does not infer time-varying active masks. Omit `tube_ids` to use the union of all track bboxes. Bboxes are in original pixel coordinates; no resize correction is applied.

Direct ROI bboxes or 64×64 block selections:

```yaml
roi:
  bboxes: [[128, 256, 1280, 720]]
  # Alternatively: blocks: [[4, 2], [4, 3]]
  # Alternatively: block_mask: a binary [block_rows, block_cols] matrix
  frames:
    "1":
      blocks: [[4, 2], [4, 3]]
```

Or `roi: {json_file: /path/to/roi.json}` with the same structure. A frame-specific entry replaces the shared entry. Multiple boxes/masks/blocks within an entry are unioned. A bbox selects a block when its center is inside the bbox, using half-open upper boundaries. An empty ROI is valid and produces zero area/coverage when positive information exists, with undefined ROI mean reported as `null`.

Saved per-layer and per-frame metrics, always together:

- ROI block ratio `R_ROI`;
- ROI information coverage `C_ROI = sum_{b in ROI} max(I_b,0) / sum_all max(I_b,0)`;
- information enrichment `E_ROI = C_ROI / R_ROI`;
- ROI / non-ROI positive densities and `D_ratio`;
- signed densities kept as auxiliary fields;
- the same metrics on each matched event, on `I_max`, and on the auxiliary event-mean map.

`I_b` is the event map, `I_max`, or event-mean depending on the row. If there is no positive importance, coverage/enrichment are `null`, not a fabricated zero. Video-level CSV rows average defined frame values; they do not pool blocks across frames.

## Outputs

Under `output.dir`:

- `frame_event_map.json`: copy of the matching file used;
- `manifest.json`: per-frame matched/uncertain events, prompt, mapping, ROI;
- `frame_XX_full_caption_scores.json`: teacher-forced scores for that frame's matched sentences;
- `layer_XX/frame_XX_blocks.csv`, `..._blocks.json`: block metadata and signed event/max/mean importance;
- `layer_XX/frame_XX_summary.json`: baseline, ROI metrics, spatial statistics, numerical checks;
- `layer_XX/frame_XX_overview.png`: original, ROI overlay, 64×64 `I_max` heatmap, heatmap+ROI boundary;
- `layer_XX/frame_XX_event_N_heatmap.png`, `..._max_heatmap.png`, `..._mean_heatmap.png`;
- `frame_XX_event_N_layer_comparison.png`, `..._max_layer_comparison.png`, `..._mean_layer_comparison.png`;
- `summary.csv` / `summary.json`: frame rows and video means with `ROI_block_ratio`, `ROI_information_coverage`, `information_enrichment`, densities, and `density_ratio`;
- `progress.json`: completed frames for long runs.

Layers default to `[0,4,8]`. No layer is auto-selected from one frame.

## Validation and constraints

The adapter supports the installed Transformers dense Qwen3-VL decoder with SDPA/eager, a single unpadded example, and all model weights on one device. No CPU/disk offload or multi-device dispatch. Batch size is configurable; lower it if memory is insufficient.

Every actual run verifies retain-all suffix replay against the original full forward and the first batched intervention against an individual intervention. It stops on excessive numerical error rather than publishing unvalidated maps. Log normalization is float32 even with BF16 weights. Reported validation errors indicate numerical precision, not scientific accuracy.

```bash
.vlm-inf/bin/python -m unittest discover -s tests -p 'test_token_information.py' -v
```

Tests cover spatial grouping across frames, partial edges, sentence token alignment, signed ROI formulas, undefined denominators, retain-all replay, DeepStack handling, and agreement with an independent full-forward pruning implementation on a tiny randomly initialized Qwen3-VL model. Only that small regression test reruns a full model for reference; the actual 8B analysis never does so per block.
