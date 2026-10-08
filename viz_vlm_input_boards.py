#!/usr/bin/env python3
"""Lay current VLM input frames onto a Global / Tube board.

Reads ``outputs/input_order/*__timestamp.txt`` dumps so the thumbnails are
the exact images sent to the model (plain Global / ROI crops).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DUMP_RE = re.compile(
    r"^\[(?P<label>Global|ROI(?P<roi>\d+))\]\s+t=(?P<time>\S+?)\s+->\s+(?P<path>.+)$"
)
DUMP_NAME_RE = re.compile(
    r"^(?P<video_id>VIRAT_.+)__(?P<method>.+)__(?P<order>timestamp|grouped|global_anchor|tube_story)\.txt$"
)

CURRENT_METHODS = (
    "full_1fps",
    "global_0.25fps_proposed_multi_roi",
    "single_motion_roi_original",
    "global_0.25fps_single_motion_roi",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dump_dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "input_order",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "vlm_input_boards",
    )
    p.add_argument("--video_id", type=str, nargs="*", default=None)
    p.add_argument("--method", type=str, nargs="*", default=list(CURRENT_METHODS))
    p.add_argument("--input_order", type=str, default="timestamp")
    p.add_argument("--thumb_h", type=int, default=72)
    p.add_argument("--frame_gap", type=int, default=8)
    p.add_argument("--row_gap", type=int, default=10)
    p.add_argument("--section_gap", type=int, default=36)
    p.add_argument("--label_w", type=int, default=150)
    p.add_argument("--pad", type=int, default=24)
    p.add_argument("--canvas_w", type=int, default=2400)
    p.add_argument("--section_pad", type=int, default=12)
    p.add_argument(
        "--layout",
        choices=["grouped", "order", "both"],
        default="grouped",
        help="grouped = Global/ROI rows; order = actual VLM sequence; both = write two PNGs",
    )
    return p.parse_args()


def parse_dump_ordered(path: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = DUMP_RE.match(line)
        if m is None:
            continue
        label = m.group("label")
        time_s = m.group("time")
        img = Path(m.group("path").strip())
        if label == "Global":
            text = f"[Global] t={time_s}s"
        else:
            text = f"[ROI{m.group('roi')}] t={time_s}s"
        items.append((text, img))
    return items


def parse_dump(path: Path) -> tuple[list[Path], dict[int, list[Path]]]:
    return split_items_grouped(parse_dump_ordered(path))


def split_items_grouped(
    items: list[tuple[str, Path]],
) -> tuple[list[Path], dict[int, list[Path]]]:
    globals_: list[Path] = []
    tubes: dict[int, list[Path]] = defaultdict(list)
    for label, img in items:
        if label.startswith("[Global]"):
            globals_.append(img)
            continue
        m = re.match(r"\[ROI(\d+)\]", label)
        if m:
            tubes[int(m.group(1))].append(img)
    return globals_, dict(tubes)


def default_board_args(**overrides: object) -> argparse.Namespace:
    args = argparse.Namespace(
        thumb_h=72,
        frame_gap=8,
        row_gap=10,
        section_gap=36,
        label_w=170,
        pad=24,
        canvas_w=2400,
        section_pad=12,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def render_grouped_canvas(
    items: list[tuple[str, Path]],
    *,
    video_id: str,
    method: str,
    args: argparse.Namespace | None = None,
    tube_labels: dict[int, str] | None = None,
    extra_sections: list[tuple[str, list[Path]]] | None = None,
) -> np.ndarray:
    args = args or default_board_args()
    globals_, tubes = split_items_grouped(items)
    sections: list[tuple[str, list[Path]]] = []
    if globals_:
        sections.append((global_section_name(method), globals_))
    for tid in sorted(tubes):
        label = (tube_labels or {}).get(tid) or f"Tube {tid:02d}  (ROI crop)"
        sections.append((label, tubes[tid]))
    for extra_name, extra_paths in extra_sections or []:
        if extra_paths:
            sections.append((extra_name, extra_paths))
    if not sections:
        raise RuntimeError("no frames to render")
    return render_sections(
        sections,
        title=f"{video_id}  |  {method}  |  grouped",
        canvas_w=int(args.canvas_w),
        label_w=int(args.label_w),
        pad=int(args.pad),
        thumb_h=int(args.thumb_h),
        frame_gap=int(args.frame_gap),
        row_gap=int(args.row_gap),
        section_gap=int(args.section_gap),
        section_pad=int(args.section_pad),
    )


def _caption_from_label(label: str) -> str:
    if label.startswith("[Global]"):
        rest = label.replace("[Global]", "").strip()
        return f"Global {rest}"
    return label.replace("[", "").replace("]", "")


def render_order_canvas(
    items: list[tuple[str, Path]],
    *,
    video_id: str,
    method: str,
    args: argparse.Namespace | None = None,
) -> np.ndarray:
    """One wrapping strip in actual VLM input order, with Global / ROI labels."""
    args = args or default_board_args()
    if not items:
        raise RuntimeError("no frames to render")
    thumb_h = int(args.thumb_h)
    cap_h = 22
    labeled = load_labeled_thumbs(items, thumb_h)
    cards: list[np.ndarray] = []
    for i, (label, thumb) in enumerate(labeled):
        h, w = thumb.shape[:2]
        card = np.full((h + cap_h, w, 3), 252, dtype=np.uint8)
        card[:h] = thumb
        caption = f"{i:02d} {_caption_from_label(label)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        (tw, th), _ = cv2.getTextSize(caption, font, scale, 1)
        tx = max(2, (w - tw) // 2)
        color = kind_border_color(label)
        cv2.putText(card, caption, (tx, h + 15), font, scale, color, 1, cv2.LINE_AA)
        cards.append(card)

    plot_w = int(args.canvas_w) - int(args.label_w) - 2 * int(args.pad)
    rows = wrap_rows(cards, plot_w=plot_w, frame_gap=int(args.frame_gap))
    packed = [("VLM input order", rows, (236, 240, 245))]
    section_heights = []
    for _name, rowset, _color in packed:
        row_hs = [max(th.shape[0] for _, th in row) for row in rowset]
        h = 2 * int(args.section_pad) + sum(row_hs) + int(args.row_gap) * (len(rowset) - 1)
        section_heights.append(h)

    title_h = 72
    pad = int(args.pad)
    canvas_w = int(args.canvas_w)
    canvas_h = title_h + pad + section_heights[0] + pad
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :, 0] = np.clip(226 + (yy * 0.015).astype(np.int32), 0, 255)
    canvas[:, :, 1] = np.clip(228 + (xx * 0.006).astype(np.int32), 0, 255)
    canvas[:, :, 2] = np.clip(232 - (yy * 0.01).astype(np.int32), 0, 255)

    cv2.putText(
        canvas,
        f"{video_id}  |  {method}  |  VLM order",
        (pad, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Blue border = Global (full-frame)    Green border = ROI crop",
        (pad, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    x_origin = int(args.label_w) + pad
    y = title_h + pad
    color = packed[0][2]
    sec_h = section_heights[0]
    cv2.rectangle(canvas, (x_origin - 10, y), (canvas_w - pad, y + sec_h), color, -1)
    cv2.rectangle(canvas, (x_origin - 10, y), (canvas_w - pad, y + sec_h), (195, 200, 205), 1)
    name = packed[0][0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_tw, th), _ = cv2.getTextSize(name, font, 0.5, 1)
    cv2.putText(canvas, name, (pad, y + (sec_h + th) // 2), font, 0.5, (45, 45, 45), 1, cv2.LINE_AA)
    row_y = y + int(args.section_pad)
    for row in packed[0][1]:
        row_h = max(th.shape[0] for _, th in row)
        for x_off, thumb in row:
            paste(canvas, thumb, x_origin + x_off, row_y)
        row_y += row_h + int(args.row_gap)
    return canvas


def global_section_name(method: str) -> str:
    if "accum" in method or "accumulated" in method:
        return "Global  0.25fps"
    if "video_start" in method:
        return "Global  video start"
    if "start_mid_end" in method:
        return "Global  start+mid+end"
    if "start_end" in method:
        return "Global  start+end"
    if "start_only" in method:
        return "Global  start"
    if method.startswith("full_1fps") or "1fps" in method and "0.25" not in method:
        return "Global  1fps"
    if "2fps" in method:
        return "Global  2fps"
    if "0.25fps" in method:
        return "Global  0.25fps"
    return "Global"


def resize_thumb(bgr: np.ndarray, thumb_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("empty image")
    scale = float(thumb_h) / float(h)
    nw = max(1, int(round(w * scale)))
    return cv2.resize(bgr, (nw, int(thumb_h)), interpolation=cv2.INTER_AREA)


def card_with_border(
    thumb: np.ndarray,
    border: int = 2,
    color: tuple[int, int, int] = (180, 185, 190),
) -> np.ndarray:
    h, w = thumb.shape[:2]
    out = np.full((h + 2 * border, w + 2 * border, 3), 250, dtype=np.uint8)
    out[border : border + h, border : border + w] = thumb
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, border)
    return out


def paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + pw), min(ch, y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    canvas[y0:y1, x0:x1] = patch[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]


GLOBAL_BORDER = (196, 126, 46)  # BGR blue-ish
ROI_BORDER = (62, 150, 72)  # BGR green


def kind_border_color(label: str) -> tuple[int, int, int]:
    if label.startswith("[Global]") or label.startswith("Global"):
        return GLOBAL_BORDER
    return ROI_BORDER


def load_thumbs(paths: list[Path], thumb_h: int) -> list[np.ndarray]:
    thumbs: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"failed to read {path}")
        thumbs.append(card_with_border(resize_thumb(img, thumb_h), border=1))
    return thumbs


def load_labeled_thumbs(
    items: list[tuple[str, Path]],
    thumb_h: int,
) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for label, path in items:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"failed to read {path}")
        color = kind_border_color(label)
        out.append((label, card_with_border(resize_thumb(img, thumb_h), border=2, color=color)))
    return out


def wrap_rows(
    thumbs: list[np.ndarray],
    *,
    plot_w: int,
    frame_gap: int,
) -> list[list[tuple[int, np.ndarray]]]:
    rows: list[list[tuple[int, np.ndarray]]] = []
    if not thumbs:
        return rows
    cur: list[tuple[int, np.ndarray]] = []
    x = 0
    for thumb in thumbs:
        w = int(thumb.shape[1])
        if cur and x + w > plot_w:
            rows.append(cur)
            cur = []
            x = 0
        cur.append((x, thumb))
        x += w + int(frame_gap)
    if cur:
        rows.append(cur)
    return rows


def section_band_color(index: int) -> tuple[int, int, int]:
    colors = (
        (236, 240, 245),
        (232, 239, 234),
        (239, 234, 240),
        (234, 238, 242),
        (237, 236, 232),
    )
    return colors[index % len(colors)]


def render_sections(
    sections: list[tuple[str, list[Path]]],
    *,
    title: str,
    canvas_w: int,
    label_w: int,
    pad: int,
    thumb_h: int,
    frame_gap: int,
    row_gap: int,
    section_gap: int,
    section_pad: int,
) -> np.ndarray:
    plot_w = canvas_w - label_w - 2 * pad
    title_h = 56

    packed: list[tuple[str, list[list[tuple[int, np.ndarray]]], tuple[int, int, int]]] = []
    for i, (name, paths) in enumerate(sections):
        thumbs = load_thumbs(paths, thumb_h)
        rows = wrap_rows(thumbs, plot_w=plot_w, frame_gap=frame_gap)
        packed.append((name, rows, section_band_color(i)))

    section_heights: list[int] = []
    for _name, rows, _color in packed:
        if not rows:
            section_heights.append(2 * section_pad + thumb_h)
            continue
        row_hs = [max(th.shape[0] for _, th in row) for row in rows]
        h = 2 * section_pad + sum(row_hs) + row_gap * (len(rows) - 1)
        section_heights.append(h)

    canvas_h = (
        title_h
        + pad
        + sum(section_heights)
        + section_gap * max(0, len(section_heights) - 1)
        + pad
    )

    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :, 0] = np.clip(226 + (yy * 0.015).astype(np.int32), 0, 255)
    canvas[:, :, 1] = np.clip(228 + (xx * 0.006).astype(np.int32), 0, 255)
    canvas[:, :, 2] = np.clip(232 - (yy * 0.01).astype(np.int32), 0, 255)

    cv2.putText(
        canvas,
        title,
        (pad, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    x_origin = label_w + pad
    y = title_h + pad
    for sec_i, ((name, rows, color), sec_h) in enumerate(zip(packed, section_heights)):
        cv2.rectangle(
            canvas,
            (x_origin - 10, y),
            (canvas_w - pad, y + sec_h),
            color,
            thickness=-1,
        )
        cv2.rectangle(
            canvas,
            (x_origin - 10, y),
            (canvas_w - pad, y + sec_h),
            (195, 200, 205),
            1,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (_tw, th), _ = cv2.getTextSize(name, font, scale, 1)
        ty = y + (sec_h + th) // 2
        cv2.putText(canvas, name, (pad, ty), font, scale, (45, 45, 45), 1, cv2.LINE_AA)

        row_y = y + section_pad
        for row in rows:
            row_h = max(th.shape[0] for _, th in row) if row else thumb_h
            for x_off, thumb in row:
                paste(canvas, thumb, x_origin + x_off, row_y)
            row_y += row_h + row_gap

        y += sec_h
        if sec_i < len(packed) - 1:
            y += section_gap

    return canvas


def render_one(dump_path: Path, video_id: str, method: str, args: argparse.Namespace) -> Path:
    items = parse_dump_ordered(dump_path)
    if not items:
        raise RuntimeError(f"no frames in {dump_path}")
    layout = str(getattr(args, "layout", "grouped")).strip().lower()
    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if layout in {"grouped", "both"}:
        canvas = render_grouped_canvas(items, video_id=video_id, method=method, args=args)
        out_path = out_dir / f"{video_id}__{method}.png"
        if not cv2.imwrite(str(out_path), canvas):
            raise RuntimeError(f"failed to write {out_path}")
        written.append(out_path)
        print(
            f"wrote {out_path} | frames={len(items)} "
            f"size={canvas.shape[1]}x{canvas.shape[0]}"
        )
    if layout in {"order", "both"}:
        canvas = render_order_canvas(items, video_id=video_id, method=method, args=args)
        suffix = "" if layout == "order" else "__order"
        out_path = out_dir / f"{video_id}__{method}{suffix}.png"
        if not cv2.imwrite(str(out_path), canvas):
            raise RuntimeError(f"failed to write {out_path}")
        written.append(out_path)
        print(
            f"wrote {out_path} | frames={len(items)} "
            f"size={canvas.shape[1]}x{canvas.shape[0]}"
        )
    if not written:
        raise ValueError(f"unknown layout '{layout}' (use grouped, order, both)")
    return written[0]


def list_dumps(args: argparse.Namespace) -> list[tuple[Path, str, str]]:
    wanted_methods = set(args.method)
    wanted_videos = set(args.video_id) if args.video_id else None
    rows: list[tuple[Path, str, str]] = []
    for path in sorted(args.dump_dir.glob("*.txt")):
        m = DUMP_NAME_RE.match(path.name)
        if m is None:
            continue
        if m.group("order") != args.input_order:
            continue
        video_id = m.group("video_id")
        method = m.group("method")
        if method not in wanted_methods:
            continue
        if wanted_videos is not None and video_id not in wanted_videos:
            continue
        rows.append((path, video_id, method))
    return rows


def main() -> int:
    args = parse_args()
    jobs = list_dumps(args)
    if not jobs:
        raise SystemExit(f"no input_order dumps under {args.dump_dir}")
    failed: list[str] = []
    for i, (dump_path, video_id, method) in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {video_id} / {method}")
        try:
            render_one(dump_path, video_id, method, args)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {video_id} {method}: {exc}")
            failed.append(f"{video_id}__{method}")
    print(f"done ok={len(jobs) - len(failed)} failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
