#!/usr/bin/env python3
"""Pre-validation of the 40,960-token setup: does video.max_pixels/total_pixels act in the existing input path?"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
from qwen_vl_utils.vision_process import fetch_video, smart_resize

F = 32
def toks(n, h, w): return math.ceil(n / 2) * (h // F) * (w // F)

def existing_path(n, w, h, **opts):
    frames = [Image.fromarray(np.zeros((h, w, 3), np.uint8)) for _ in range(n)]
    item = {"type": "video", "video": frames, "sample_fps": 2.0, **opts}
    v = fetch_video(item, image_patch_size=16)
    v = v[0] if isinstance(v, tuple) else v
    t = v.shape[0]; H, W = int(v.shape[2]), int(v.shape[3])
    return t, H, W, toks(t, H, W)

NEW = dict(min_pixels=4096, max_pixels=786432, total_pixels=41943040)
OLD = dict(min_pixels=4096, max_pixels=262144, total_pixels=20971520)  # git HEAD configs/default.yaml
print("## 1. existing path (qwen_vl_utils.fetch_video, same options model.py _apply_video_opts passes)")
for name, o in (("new", NEW), ("old(HEAD,20480)", OLD)):
    for n, w, h in ((15, 1280, 720), (60, 1280, 720), (120, 1280, 720), (20, 1920, 1080), (160, 1920, 1080)):
        t, H, W, tk = existing_path(n, w, h, **o)
        print(f"{name:16s} frames={n:4d} native={w}x{h} -> {W}x{H}  tokens={tk}")
print("\n## per-item vs joint: two 60-frame 1280x720 videos, new options")
t, H, W, tk = existing_path(60, 1280, 720, **NEW)
print(f"each item -> {W}x{H} tokens={tk}; two items = {2*tk} (> 40960 if no joint allocation)")
print("\n## 2. 1-5 plan (budget 20480, native-first search) vs old default.yaml cap 262144 per frame")
for clip, (w, h) in {"A": (1280, 720), "C": (1280, 720), "D": (1920, 1080), "E": (1280, 720)}.items():
    for fps, n in ((0.5, 12), (1.0, 23)):
        plan_h = math.ceil(h / 32) * 32; plan_w = math.ceil(w / 32) * 32
        H, W = smart_resize(h, w, 32, 4096, 262144)
        print(f"{clip} {fps}fps native-padded {plan_w}x{plan_h} tokens={toks(n, plan_h, plan_w)} | old cap -> {W}x{H} tokens={toks(n, H, W)}")
