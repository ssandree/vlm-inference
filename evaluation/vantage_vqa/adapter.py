"""VANTAGE-Bench VQA adapter.

Reproduces official TSV construction and the VANTAGE_VQA_8frame_200 subset
from source code. Does not change official option counts, file order, or
sampling. Does not run Qwen, motion analysis, or ROI generation.

Official sources (do not invent alternatives):
- TSV construction: PhysicalAI-VANTAGE-Bench ``scripts/run_lmudata.py``
  ``_vqa_process_item`` / ``_prep_vqa`` (sorted glob, dedup, sort, index).
- 200-sample: Clemson-Capstone/VANTAGE-Bench
  ``vlmeval/dataset/video_dataset_config.py`` registers
  ``VANTAGE_VQA_8frame_200`` as
  ``partial(VANTAGE_VQA, dataset='VANTAGE_VQA', nframe=8, limit=200, random_state=42)``.
- Sampling: ``vlmeval/dataset/vantage_vqa.py`` ``VANTAGE_VQA.__init__``:
  after a video-existence filter, ``DataFrame.sample(n=200, random_state=42)``.

Known official-internal difference (not "fixed" here):
- ``run_lmudata.py`` reads annotation JSON in ``sorted(glob('*.json'))`` order.
- ``vantage_vqa.py`` ``_generate_tsv_from_annotations`` uses a hardcoded file
  list. Participant TSV comes from ``run_lmudata.py``, so this adapter follows
  that path.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_DATA_ROOT = Path("/data/datasets/VANTAGE-Bench")
ANN_DIR_REL = Path("data/vqa/data_jsons/annotations")
VIDEO_DIR_REL = Path("data/vqa/videos")

# Official VANTAGE_VQA_8frame_200 (video_dataset_config.py + vantage_vqa.py).
OFFICIAL_200_LIMIT = 200
OFFICIAL_200_RANDOM_STATE = 42
PILOT_SIZE = 50

# Fallback TSV generator in vantage_vqa.py (NOT used for the 200-sample; recorded only).
LOADER_FALLBACK_FILE_ORDER = (
    "metrics_spatial_wo_ss.json",
    "VANTAGE_VQA_Verification_Final_ITS_Data.json",
    "metrics_spatial_ss.json",
    "metrics_temporal_filtered_ss.json",
    "metrics_temporal_wo_ss.json",
)


@dataclass
class VantageVqaItem:
    index: int
    video: str
    video_id: str
    q_uid: str
    question: str
    options: List[str]
    video_path: str
    source_file: str
    pool200_rank: Optional[int] = None
    n_options: int = 0

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "video_path": self.video_path,
            "video_id": self.video_id,
            "q_uid": self.q_uid,
            "question": self.question,
            "options": list(self.options),
            "video": self.video,
            "source_file": self.source_file,
            "pool200_rank": self.pool200_rank,
        }


@dataclass
class AnnotationStats:
    n_raw: int
    n_tsv: int
    n_unique_videos: int
    n_videos_on_disk: int
    per_file: Dict[str, int]
    n_options_hist: Dict[int, int]
    n_malformed_option_count: int
    n_missing_video: int
    n_empty_question: int
    n_empty_q_uid: int
    n_dedup_skipped: int
    n_duplicate_index: int
    q_uid_suffix_hist: Dict[str, int]
    videos_in_multiple_files: int
    glob_file_order: List[str]


@dataclass
class ValidationReport:
    n_items: int
    n_unique_index: int
    n_unique_videos: int
    missing_videos: List[str]
    malformed_options: List[int]
    duplicate_indices: List[int]
    videos_with_multiple_qa: Dict[str, int]
    ok: bool


def _strip_video_ext(name: str) -> str:
    """Match run_lmudata._strip_video_ext / vantage_vqa._process_annotation_item."""
    base = os.path.basename(name)
    for ext in (".json", ".mp4"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def _ensure_mp4(name: str) -> str:
    """Match run_lmudata._ensure_mp4 (TSV ``video`` field includes .mp4)."""
    return name if name.endswith(".mp4") else name + ".mp4"


def _strip_option_label(opt: Any) -> str:
    """Match run_lmudata / vantage_vqa: split once on ': '."""
    if isinstance(opt, str):
        parts = opt.split(": ", 1)
        return parts[1] if len(parts) == 2 else opt
    return str(opt)


def _q_uid_suffix(q_uid: str) -> str:
    if q_uid.endswith(".json"):
        return ".json"
    if q_uid.endswith(".mp4"):
        return ".mp4"
    return "other"


def _process_annotation_item(item: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
    """Match run_lmudata._vqa_process_item, keeping raw question and original q_uid."""
    raw_uid = item.get("q_uid") or item.get("vid") or ""
    if not raw_uid:
        return None
    video_id = _strip_video_ext(raw_uid)
    video = _ensure_mp4(video_id)
    question = item.get("question", "")
    if not question:
        return None
    raw_opts = item.get("options", []) or []
    options = [_strip_option_label(opt) for opt in raw_opts]
    if not options:
        return None
    return {
        "video": video,
        "video_id": video_id,
        "q_uid": raw_uid,
        "question": question,
        "options": options,
        "n_options": len(options),
        "_qid": item.get("question_id", ""),
        "source_file": source_file,
    }


def annotation_dir(data_root: Path) -> Path:
    return Path(data_root) / ANN_DIR_REL


def video_dir(data_root: Path) -> Path:
    return Path(data_root) / VIDEO_DIR_REL


def resolve_video_path(data_root: Path, video: str) -> Path:
    """Match vantage_vqa video-existence check: stem + '.mp4' under videos/."""
    name = str(video).removesuffix(".mp4") + ".mp4"
    return video_dir(data_root) / name


def inspect_raw_annotations(data_root: Path = DEFAULT_DATA_ROOT) -> AnnotationStats:
    ann = annotation_dir(data_root)
    vdir = video_dir(data_root)
    videos_on_disk = {p.name for p in vdir.glob("*.mp4")}
    glob_files = [p.name for p in sorted(ann.glob("*.json"))]

    n_raw = 0
    per_file: Dict[str, int] = {}
    n_options_hist: Counter[int] = Counter()
    n_empty_question = 0
    n_empty_q_uid = 0
    suffix_hist: Counter[str] = Counter()
    vid_files: Dict[str, set] = defaultdict(set)

    for jf in sorted(ann.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        per_file[jf.name] = len(data)
        n_raw += len(data)
        for item in data:
            q_uid = item.get("q_uid") or ""
            if not q_uid:
                n_empty_q_uid += 1
            if not item.get("question"):
                n_empty_question += 1
            opts = item.get("options")
            n_options_hist[len(opts) if isinstance(opts, list) else -1] += 1
            suffix_hist[_q_uid_suffix(q_uid)] += 1
            if q_uid:
                vid_files[_ensure_mp4(_strip_video_ext(q_uid))].add(jf.name)

    return AnnotationStats(
        n_raw=n_raw,
        n_tsv=0,
        n_unique_videos=len(vid_files),
        n_videos_on_disk=len(videos_on_disk),
        per_file=per_file,
        n_options_hist={int(k): int(v) for k, v in sorted(n_options_hist.items())},
        n_malformed_option_count=sum(v for k, v in n_options_hist.items() if k != 4),
        n_missing_video=0,
        n_empty_question=n_empty_question,
        n_empty_q_uid=n_empty_q_uid,
        n_dedup_skipped=0,
        n_duplicate_index=0,
        q_uid_suffix_hist=dict(suffix_hist),
        videos_in_multiple_files=sum(1 for files in vid_files.values() if len(files) > 1),
        glob_file_order=glob_files,
    )


def build_official_tsv_rows(data_root: Path = DEFAULT_DATA_ROOT) -> Tuple[List[Dict[str, Any]], AnnotationStats]:
    """Reproduce run_lmudata._prep_vqa row construction (no TSV write)."""
    stats = inspect_raw_annotations(data_root)
    ann = annotation_dir(data_root)

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    n_dedup = 0
    for jf in sorted(ann.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for item in data:
            proc = _process_annotation_item(item, jf.name)
            if proc is None:
                continue
            dedup_key = (proc["video"], proc["_qid"] or proc["question"][:80])
            if dedup_key in seen:
                n_dedup += 1
                continue
            seen.add(dedup_key)
            rows.append(proc)

    rows.sort(key=lambda r: (r["video"], r["_qid"]))
    for i, r in enumerate(rows):
        r["index"] = i
        r.pop("_qid", None)

    indices = [r["index"] for r in rows]
    n_dup_index = len(indices) - len(set(indices))
    unique_videos = {r["video"] for r in rows}
    missing = [r for r in rows if not resolve_video_path(data_root, r["video"]).is_file()]

    stats.n_tsv = len(rows)
    stats.n_unique_videos = len(unique_videos)
    stats.n_missing_video = len(missing)
    stats.n_dedup_skipped = n_dedup
    stats.n_duplicate_index = n_dup_index
    return rows, stats


def load_vantage_vqa_items(data_root: Path = DEFAULT_DATA_ROOT) -> Tuple[List[VantageVqaItem], AnnotationStats]:
    rows, stats = build_official_tsv_rows(data_root)
    items: List[VantageVqaItem] = []
    for r in rows:
        path = resolve_video_path(data_root, r["video"])
        items.append(
            VantageVqaItem(
                index=int(r["index"]),
                video=r["video"],
                video_id=r["video_id"],
                q_uid=r["q_uid"],
                question=r["question"],
                options=list(r["options"]),
                video_path=str(path),
                source_file=r["source_file"],
                n_options=int(r["n_options"]),
            )
        )
    return items, stats


def filter_existing_videos(items: Sequence[VantageVqaItem]) -> List[VantageVqaItem]:
    """Match vantage_vqa.VANTAGE_VQA.__init__ video-existence filter (keep order)."""
    kept = [it for it in items if Path(it.video_path).is_file()]
    return kept


def official_200_pool(
    items: Sequence[VantageVqaItem],
    *,
    limit: int = OFFICIAL_200_LIMIT,
    random_state: int = OFFICIAL_200_RANDOM_STATE,
) -> List[VantageVqaItem]:
    """Reproduce VANTAGE_VQA_8frame_200: pandas sample after video filter.

    Official code (vantage_vqa.py)::

        sample_num = min(int(self.limit), len(self.data))
        self.data = self.data.sample(n=sample_num, random_state=self.random_state)

    ``limit=200`` is > 1.0, so the fractional branch is not used.
    """
    kept = filter_existing_videos(items)
    if not kept:
        return []
    records = [
        {
            "index": it.index,
            "video": it.video,
            "video_id": it.video_id,
            "q_uid": it.q_uid,
            "question": it.question,
            "options": it.options,
            "video_path": it.video_path,
            "source_file": it.source_file,
            "n_options": it.n_options,
        }
        for it in kept
    ]
    df = pd.DataFrame(records)
    sample_num = min(int(limit), len(df))
    sampled = df.sample(n=sample_num, random_state=random_state)
    out: List[VantageVqaItem] = []
    for rank, row in enumerate(sampled.to_dict("records")):
        out.append(
            VantageVqaItem(
                index=int(row["index"]),
                video=row["video"],
                video_id=row["video_id"],
                q_uid=row["q_uid"],
                question=row["question"],
                options=list(row["options"]),
                video_path=row["video_path"],
                source_file=row["source_file"],
                pool200_rank=rank,
                n_options=int(row["n_options"]),
            )
        )
    return out


def pilot_50_from_pool(
    pool200: Sequence[VantageVqaItem],
    *,
    n: int = PILOT_SIZE,
) -> List[VantageVqaItem]:
    """Pilot 50 rule (not an official VANTAGE key).

    Take the first ``n`` rows of the official 200-sample in pandas
    ``DataFrame.sample`` order. No extra RNG, no re-sort, no category
    stratification. This is a prefix of VANTAGE_VQA_8frame_200.
    """
    if n > len(pool200):
        raise ValueError(f"requested {n} pilot items but pool has {len(pool200)}")
    return list(pool200[:n])


def select_pilot50(data_root: Path = DEFAULT_DATA_ROOT) -> Tuple[List[VantageVqaItem], List[VantageVqaItem], AnnotationStats]:
    items, stats = load_vantage_vqa_items(data_root)
    pool200 = official_200_pool(items)
    pilot = pilot_50_from_pool(pool200)
    return pilot, pool200, stats


def validate_pilot_items(items: Sequence[VantageVqaItem]) -> ValidationReport:
    indices = [it.index for it in items]
    dup = [idx for idx, c in Counter(indices).items() if c > 1]
    missing = [it.video_path for it in items if not Path(it.video_path).is_file()]
    malformed = [it.index for it in items if len(it.options) != 4]
    vid_counts = Counter(it.video_id for it in items)
    multi = {k: int(v) for k, v in vid_counts.items() if v > 1}
    ok = (not missing) and (not malformed) and (not dup) and (len(indices) == len(set(indices)))
    return ValidationReport(
        n_items=len(items),
        n_unique_index=len(set(indices)),
        n_unique_videos=len(vid_counts),
        missing_videos=missing,
        malformed_options=malformed,
        duplicate_indices=dup,
        videos_with_multiple_qa=multi,
        ok=ok,
    )


def selection_meta(
    stats: AnnotationStats,
    pool200: Sequence[VantageVqaItem],
    pilot: Sequence[VantageVqaItem],
    report: ValidationReport,
    data_root: Path,
) -> Dict[str, Any]:
    pool_vid = Counter(it.video_id for it in pool200)
    return {
        "dataset": "VANTAGE_VQA",
        "data_root": str(data_root),
        "annotation_dir": str(annotation_dir(data_root)),
        "video_dir": str(video_dir(data_root)),
        "official_variant": "VANTAGE_VQA_8frame_200",
        "official_200_selection": {
            "source_files": [
                "Clemson-Capstone/VANTAGE-Bench/vlmeval/dataset/video_dataset_config.py",
                "Clemson-Capstone/VANTAGE-Bench/vlmeval/dataset/vantage_vqa.py",
            ],
            "registration": (
                "partial(VANTAGE_VQA, dataset='VANTAGE_VQA', nframe=8, "
                "limit=200, random_state=42)"
            ),
            "implementation": (
                "After video-existence filter, "
                "pandas.DataFrame.sample(n=min(200, N), random_state=42). "
                "limit=200 uses the integer branch (not the fractional limit<=1.0 branch)."
            ),
            "tsv_construction": (
                "PhysicalAI-VANTAGE-Bench scripts/run_lmudata.py _prep_vqa: "
                "sorted(glob *.json), skip non-lists, _vqa_process_item, "
                "dedup (video, question_id or question[:80]), "
                "sort (video, qid), assign sequential index."
            ),
            "n": len(pool200),
            "n_unique_videos": len(pool_vid),
            "random_state": OFFICIAL_200_RANDOM_STATE,
            "limit": OFFICIAL_200_LIMIT,
            "indices_in_sample_order": [it.index for it in pool200],
        },
        "pilot_50_selection": {
            "rule": (
                "Prefix of the official VANTAGE_VQA_8frame_200 pool: the first 50 "
                "rows in pandas DataFrame.sample order. No extra seed, no re-sort, "
                "no class/video stratification."
            ),
            "n": len(pilot),
            "n_unique_videos": report.n_unique_videos,
        },
        "full_dataset": {
            "n_raw_qa": stats.n_raw,
            "n_tsv_qa": stats.n_tsv,
            "n_unique_videos": stats.n_unique_videos,
            "n_videos_on_disk": stats.n_videos_on_disk,
            "per_file": stats.per_file,
            "n_options_hist": stats.n_options_hist,
            "n_malformed_option_count": stats.n_malformed_option_count,
            "n_missing_video": stats.n_missing_video,
            "n_empty_question": stats.n_empty_question,
            "n_empty_q_uid": stats.n_empty_q_uid,
            "n_dedup_skipped": stats.n_dedup_skipped,
            "n_duplicate_index": stats.n_duplicate_index,
            "q_uid_suffix_hist": stats.q_uid_suffix_hist,
            "videos_in_multiple_annotation_files": stats.videos_in_multiple_files,
            "run_lmudata_glob_file_order": stats.glob_file_order,
            "loader_fallback_file_order": list(LOADER_FALLBACK_FILE_ORDER),
        },
        "differences_from_official_not_changed": [
            (
                "run_lmudata.py glob order != vantage_vqa._generate_tsv_from_annotations "
                "hardcoded file order. Adapter follows run_lmudata (participant TSV)."
            ),
            (
                "Public annotations have only q_uid/question/options. question_id, "
                "gt_option, industry, task_type are absent; not fabricated."
            ),
            (
                "Four public items have 2 options, not 4. Official prep does not drop "
                "them. They are kept in the full TSV; none entered the seed-42 200-pool."
            ),
            (
                "ITS q_uid values end in .json; others end in .mp4. Official strip+ensure "
                "mp4 mapping is used; suffixes are not rewritten in source JSON."
            ),
            (
                "Manifest question is the raw annotation question, not the formatted TSV "
                "prompt string written by run_lmudata._vqa_format_question."
            ),
        ],
        "validation": {
            "ok": report.ok,
            "n_items": report.n_items,
            "n_unique_index": report.n_unique_index,
            "n_unique_videos": report.n_unique_videos,
            "missing_videos": report.missing_videos,
            "malformed_options_indices": report.malformed_options,
            "duplicate_indices": report.duplicate_indices,
            "videos_with_multiple_qa": report.videos_with_multiple_qa,
        },
    }


def build_pilot50_manifest(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Dict[str, Any]:
    pilot, pool200, stats = select_pilot50(data_root)
    report = validate_pilot_items(pilot)
    if not report.ok:
        raise RuntimeError(
            "pilot50 validation failed: "
            f"missing={report.missing_videos} malformed={report.malformed_options} "
            f"dup_index={report.duplicate_indices}"
        )
    return {
        "meta": selection_meta(stats, pool200, pilot, report, Path(data_root)),
        "items": [it.to_manifest_dict() for it in pilot],
    }


def write_pilot50_manifest(path: Path, data_root: Path = DEFAULT_DATA_ROOT) -> Dict[str, Any]:
    payload = build_pilot50_manifest(data_root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
