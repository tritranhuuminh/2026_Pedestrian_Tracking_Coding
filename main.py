"""
main.py
=======

YOLO26n-Pose + BoT-SORT, ablation runner (cộng dồn từng thành phần của part-appearance).

Mỗi thí nghiệm là một file YAML trong configs/. File kế thừa configs/_base.yaml
qua `extends`. Có thể bật/tắt từng component và ghi đè từng tham số ngay từ
dòng lệnh, không cần sửa file.

Components
    part_appearance   bật toàn bộ pipeline part-appearance (tracker 'ours')
    -- extractor --
    part_torso part_thigh part_shin   part nào được dùng
    luminance                         kênh L trong histogram Lab
    q_keypoint q_pixels occlusion     các hệ số của quality
    -- tracker --
    ema_smoothing                     EMA feature theo track (tắt: lấy feature mới nhất)
    ema_quality                       alpha EMA theo quality (tắt: alpha cố định)
    ema_q_gate                        chặn update EMA khi quality < feat_q_min
    match_q_gate                      lọc part của detection theo match_q_min
    shared_gate                       yêu cầu tỉ lệ part chung >= min_shared
    appearance_gate                   ngưỡng appearance_thresh (fusion 'min')
    proximity_gate                    mask proximity_thresh khi fuse với IoU
    weighted_fusion                   lam*iou+(1-lam)*emb thay cho min(iou, emb)
    -- hậu xử lý --
    interpolate                       nội suy offline

Thí nghiệm (configs/, sinh bằng tools/gen_ablation_configs.py):
    E0, E1                  original BoT-SORT (IoU / FastReID) - baseline ngoài
    S0_iou                  tracker 'ours', không part-appearance (IoU only)
    S1_torso ... S9_shared_gate
                            ABLATION CỘNG DỒN: mỗi bước thêm đúng một thành phần
                            (S9 = mô hình đầy đủ, tương đương R3 cũ)
    S10_weighted_fusion     S9 với weighted fusion
    LOO_no_<component>      leave-one-out: S9 bỏ đúng một thành phần

Ví dụ:
    python main.py --experiment S9_shared_gate
    python main.py --experiment S0_iou,S1_torso,S2_thigh --no-display   # một đoạn của chuỗi cộng dồn
    python main.py --experiment LOO_no_occlusion --no-display
    python main.py --experiment S9_shared_gate --off occlusion,q_pixels # bỏ nhiều thành phần
    python main.py --experiment S9_shared_gate --on weighted_fusion --sweep fusion_lambda=0.6,0.8,0.9
    python main.py --experiment S9_shared_gate --set min_keypoint_conf=0.5
    python main.py --experiment S9_shared_gate --sweep torso_shrink=0.6,0.7,0.8 occ_power=1.0,2.0
    python main.py --experiment all --no-display
    python main.py --list
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from tracker.bot_sort_original import BoTSORT as OriginalBoTSORT
from tracker.bot_sort_ours import BoTSORT as ExperimentalBoTSORT
from tracker.bot_sort_ours import EXTRACTOR_FLAGS, EXTRACTOR_PARAMS, TRACKER_FLAGS
from tracker.part_appearance import PartAppearanceExtractor
from visualize import visualize_frame

# ============================================================
# PROJECT PATHS / DEFAULTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = PROJECT_ROOT / "weights" / "yolo26n-pose.pt"
DEFAULT_VIDEO = PROJECT_ROOT / "assets" / "video_0076.mp4"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "configs"

DEFAULT_EXPERIMENT = "S10"

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70
DEFAULT_MAX_FRAMES = None
DEFAULT_SAVE_VIDEO = True
DEFAULT_DISPLAY = True
DEFAULT_WARMUP_FRAMES = 10

NUM_PARTS = PartAppearanceExtractor.NUM_PARTS
PART_NAMES = PartAppearanceExtractor.PART_NAMES


# ============================================================
# COMPONENT SCHEMA
# ============================================================

# Thành phần của R3 (chỉ có nghĩa khi part_appearance bật); mỗi cờ use_<name> tương ứng
# một cờ trong EXTRACTOR_FLAGS / TRACKER_FLAGS của tracker.
PART_COMPONENTS_EXTRACTOR = ("part_torso", "part_thigh", "part_shin",
                             "luminance", "q_keypoint", "q_pixels", "occlusion")
PART_COMPONENTS_TRACKER = ("ema_smoothing", "ema_quality", "ema_q_gate",
                           "match_q_gate", "shared_gate",
                           "appearance_gate", "proximity_gate")
FLAG_COMPONENTS = PART_COMPONENTS_EXTRACTOR + PART_COMPONENTS_TRACKER
PART_COMPONENTS = FLAG_COMPONENTS + ("weighted_fusion",)

COMPONENTS = ("part_appearance",) + PART_COMPONENTS + ("interpolate",)

COMPONENT_GROUPS = (
    ("PIPELINE", ("part_appearance",)),
    ("EXTRACTOR", PART_COMPONENTS_EXTRACTOR),
    ("TRACKER", PART_COMPONENTS_TRACKER + ("weighted_fusion",)),
    ("POST-PROCESSING", ("interpolate",)),
)

# Kiểm tra lúc import: cờ của tracker và danh sách component phải khớp nhau
assert {f"use_{n}" for n in PART_COMPONENTS_EXTRACTOR} == set(EXTRACTOR_FLAGS), \
    "PART_COMPONENTS_EXTRACTOR lệch với EXTRACTOR_FLAGS của bot_sort_ours"
assert {f"use_{n}" for n in PART_COMPONENTS_TRACKER} == set(TRACKER_FLAGS), \
    "PART_COMPONENTS_TRACKER lệch với TRACKER_FLAGS của bot_sort_ours"
assert tuple(f"part_{n}" for n in PART_NAMES) == PART_COMPONENTS_EXTRACTOR[:NUM_PARTS], \
    "part_* component phải theo đúng thứ tự PartAppearanceExtractor.PARTS"

TRACKER_KINDS = ("original", "ours")

# Key bắt buộc của từng section trong file config
YAML_KEYS = ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer",
             "match_thresh", "fuse_score", "proximity_thresh", "appearance_thresh", "cmc_method")
PART_KEYS = ("fusion_lambda", "part_calib", "feat_alpha", "feat_q_min",
             "match_q_min", "min_shared", "kps_normalized")
EXTRACTOR_KEYS = tuple(EXTRACTOR_PARAMS) + ("debug_reid",)
FAST_REID_KEYS = ("enabled", "config", "weights")

SECTION_KEYS = {
    "botsort": YAML_KEYS,
    "part": PART_KEYS,
    "extractor": EXTRACTOR_KEYS,
    "fast_reid": FAST_REID_KEYS,
}
REQUIRED_TOP = ("tracker", "mot20", "components", "botsort", "fast_reid", "part", "extractor")

# Tham số chỉ có tác dụng khi component tương ứng bật (kiểm tra với run dùng part-appearance).
PARAM_REQUIRES = {
    "fusion_lambda":       (lambda r: r.weighted_fusion, "weighted_fusion"),
    "appearance_thresh":   (lambda r: r.appearance_gate, "appearance_gate (và không weighted_fusion)"),
    "feat_alpha":          (lambda r: r.ema_smoothing, "ema_smoothing"),
    "feat_q_min":          (lambda r: r.ema_q_gate, "ema_q_gate"),
    "match_q_min":         (lambda r: r.match_q_gate, "match_q_gate"),
    "min_shared":          (lambda r: r.shared_gate, "shared_gate"),
    "l_weight":            (lambda r: r.luminance, "luminance"),
    "good_pixels":         (lambda r: r.q_pixels, "q_pixels"),
    "occ_power":           (lambda r: r.occlusion, "occlusion"),
    "torso_shrink":        (lambda r: r.part_torso, "part_torso"),
    "min_part_area_ratio": (lambda r: r.part_torso, "part_torso"),
    "half_w_ratio":        (lambda r: r.part_thigh or r.part_shin, "part_thigh hoặc part_shin"),
    "limb_w_ratio":        (lambda r: r.part_thigh or r.part_shin, "part_thigh hoặc part_shin"),
    "min_limb_ratio":      (lambda r: r.part_thigh or r.part_shin, "part_thigh hoặc part_shin"),
}


# ============================================================
# CONFIG LOADING
# ============================================================

def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def list_config_files(config_dir):
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục config: {config_dir}")
    return {p.stem.upper(): p for p in sorted(config_dir.glob("*.yaml"))
            if not p.stem.startswith("_")}


def validate_config(cfg, path):
    miss = [k for k in REQUIRED_TOP if k not in cfg]
    if miss:
        raise ValueError(f"{path.name}: thiếu mục {miss}")
    if cfg["tracker"] not in TRACKER_KINDS:
        raise ValueError(f"{path.name}: tracker phải là một trong {TRACKER_KINDS}, "
                         f"đang là '{cfg['tracker']}'")

    comps = cfg["components"]
    bad = [n for n in comps if n not in COMPONENTS]
    if bad:
        raise ValueError(f"{path.name}: component không hợp lệ {bad}. Hợp lệ: {', '.join(COMPONENTS)}")
    missing_c = [n for n in COMPONENTS if n not in comps]
    if missing_c:
        raise ValueError(f"{path.name}: thiếu component {missing_c} (khai báo đủ trong _base.yaml)")

    for section, expected in SECTION_KEYS.items():
        got = cfg[section]
        if not isinstance(got, dict):
            raise ValueError(f"{path.name}: mục '{section}' phải là dict")
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        if missing or extra:
            raise ValueError(f"{path.name}: mục '{section}' không khớp. "
                             f"Thiếu: {missing}, thừa: {extra}")

    fr = cfg["fast_reid"]
    if fr["enabled"] and cfg["tracker"] != "original":
        raise ValueError(f"{path.name}: fast_reid chỉ dùng được với tracker 'original'")
    if fr["enabled"] and not (fr["config"] and fr["weights"]):
        raise ValueError(f"{path.name}: fast_reid.enabled=true cần có config và weights")


def load_experiment_config(config_dir, name):
    files = list_config_files(config_dir)
    path = files.get(name.strip().upper())
    if path is None:
        raise ValueError(f"Unknown experiment '{name}'. Available: {', '.join(p.stem for p in files.values())}")

    raw = load_yaml(path)
    parent = raw.pop("extends", None)
    cfg = load_yaml(Path(config_dir) / parent) if parent else {}
    cfg = deep_merge(cfg, raw)
    validate_config(cfg, path)
    return path, cfg


# ============================================================
# RUN RESOLUTION
# ============================================================

def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def _check_components(names, what):
    bad = [n for n in names if n not in COMPONENTS]
    if bad:
        raise ValueError(f"Unknown component(s) in {what}: {bad}. Available: {', '.join(COMPONENTS)}")


def parse_kv(tokens):
    """['a=0.5', 'b=true'] -> {'a': 0.5, 'b': True} (kiểu suy ra bằng YAML)."""
    out = {}
    for t in tokens or []:
        if "=" not in t:
            raise ValueError(f"--set expects key=value, got '{t}'")
        k, v = t.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def parse_sweep(tokens):
    """['fusion_lambda=0.6,0.8'] -> list of override dicts (cartesian product)."""
    if not tokens:
        return [{}]
    keys, grids = [], []
    for t in tokens:
        if "=" not in t:
            raise ValueError(f"--sweep expects key=v1,v2, got '{t}'")
        k, v = t.split("=", 1)
        keys.append(k.strip())
        grids.append([yaml.safe_load(x) for x in v.split(",")])
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]


def resolve_run(name, config_dir, on=(), off=()):
    path, cfg = load_experiment_config(config_dir, name)
    key = path.stem
    original = cfg["tracker"] == "original"

    on, off = list(on), list(off)
    _check_components(on, "--on")
    _check_components(off, "--off")

    declared = {n: bool(cfg["components"][n]) for n in COMPONENTS}
    toggles = dict(declared)
    for n in on:
        toggles[n] = True
    for n in off:
        toggles[n] = False

    if original and toggles["part_appearance"]:
        raise ValueError(f"{key} dùng tracker original, không hỗ trợ part_appearance. "
                         f"Hãy dùng config có tracker: ours.")

    if not toggles["part_appearance"]:
        # Không có part-appearance: mọi thành phần của nó vô nghĩa -> tắt hết
        bad = [n for n in on if n in PART_COMPONENTS]
        if bad:
            raise ValueError(f"{bad} cần part_appearance bật "
                             f"(config '{key}' đang không dùng part-appearance).")
        for n in PART_COMPONENTS:
            toggles[n] = False
    else:
        # weighted_fusion thay hoàn toàn nhánh min -> appearance_gate không còn tác dụng
        if toggles["weighted_fusion"]:
            if "appearance_gate" in on:
                raise ValueError("appearance_gate chỉ dùng ở fusion 'min', không bật cùng weighted_fusion.")
            toggles["appearance_gate"] = False
        # alpha theo quality chỉ có nghĩa khi có EMA
        if not toggles["ema_smoothing"]:
            if "ema_quality" in on:
                raise ValueError("ema_quality cần ema_smoothing bật.")
            toggles["ema_quality"] = False
        if not any(toggles[f"part_{n}"] for n in PART_NAMES):
            raise ValueError(f"Cần bật ít nhất một part trong "
                             f"{[f'part_{n}' for n in PART_NAMES]}.")

    tag = key
    tag += "".join(f"+{n}" for n in on if not declared[n])
    tag += "".join(f"-{n}" for n in off if declared[n])
    return SimpleNamespace(name=key, tag=tag, config_path=path,
                           description=cfg.get("description", ""),
                           original=original, mot20=bool(cfg["mot20"]),
                           cfg=cfg, **toggles)


def uses_part(run):
    """Run có thực sự dùng part-appearance (và do đó cần nhóm tham số part/extractor)."""
    return (not run.original) and bool(run.part_appearance)


def param_unused_reason(run, key):
    """Trả về component cần bật nếu tham số `key` không có tác dụng với run này, ngược lại None."""
    if not uses_part(run):
        return None
    rule = PARAM_REQUIRES.get(key)
    if rule is None:
        return None
    ok, why = rule
    return None if ok(run) else why


def check_overrides(run, overrides):
    """Báo lỗi sớm các override sai tên hoặc không có tác dụng với run này."""
    if not overrides:
        return
    part_on = uses_part(run)
    known = set(YAML_KEYS) | (set(PART_KEYS) | set(EXTRACTOR_KEYS) if part_on else set())
    unknown = sorted(k for k in overrides if k not in known)
    if unknown:
        hint = ""
        if not part_on and any(k in PART_KEYS or k in EXTRACTOR_KEYS for k in unknown):
            hint = " (tham số part/extractor chỉ dùng khi part_appearance bật)"
        raise ValueError(f"[{run.tag}] override không hợp lệ: {unknown}{hint}. "
                         f"Hợp lệ: {sorted(known)}")
    for k in overrides:
        why = param_unused_reason(run, k)
        if why:
            raise ValueError(f"[{run.tag}] '{k}' không có tác dụng: cần bật {why}.")


def get_tracker_class(run):
    return OriginalBoTSORT if run.original else ExperimentalBoTSORT


def list_everything(config_dir):
    files = list_config_files(config_dir)
    print("Components:")
    for title, names in COMPONENT_GROUPS:
        print(f"  {title:<16}: {', '.join(names)}")
    print(f"\nExperiments ({config_dir}):")
    for name, path in files.items():
        _, cfg = load_experiment_config(config_dir, name)
        on = [n for n in COMPONENTS if cfg["components"].get(n)]
        if cfg["tracker"] == "original" or not cfg["components"]["part_appearance"]:
            desc_flags = "FastReID" if cfg["fast_reid"]["enabled"] else "IoU only"
        else:
            off = [n for n in PART_COMPONENTS if not cfg["components"][n]
                   and n not in ("weighted_fusion",)]
            desc_flags = ("part_appearance" + (f", weighted_fusion" if cfg["components"]["weighted_fusion"] else "")
                          + (f"  OFF: {', '.join(off)}" if off else ""))
        if cfg["components"]["interpolate"]:
            desc_flags += ", interpolate"
        print(f"  {path.stem:<22} {cfg['tracker']:<9} {desc_flags}"
              f"\n  {'':<22} - {cfg.get('description', '')}")
    print("\nTham số ghi đè được bằng --set / --sweep:")
    print("  botsort  :", ", ".join(YAML_KEYS))
    print("  part     :", ", ".join(PART_KEYS))
    print("  extractor:", ", ".join(EXTRACTOR_KEYS))


# ============================================================
# TRACKER CALL ADAPTER
# ============================================================
# original : update(dets, img)
# ours     : update(dets, keypoints, img)  hoặc  update(dets, img, keypoints)
# Tự nhận diện theo tên tham số; keypoints được đổi sang (N,17,3) array khi cần.

def detect_call_style(TrackerClass):
    names = [n for n in inspect.signature(TrackerClass.update).parameters if n != "self"]
    kp = [i for i, n in enumerate(names) if n.startswith("kp") or "keypoint" in n]
    if not kp:
        return "plain"
    im = [i for i, n in enumerate(names) if n in ("img", "frame", "image", "im")]
    if im and kp[0] < im[0]:
        return "kps_first"
    return "kps_last"


def keypoints_to_array(keypoints, n, k=17):
    arr = np.zeros((n, k, 3), np.float32)
    for i, kp in enumerate(keypoints):
        if kp is not None:
            arr[i, :len(kp)] = kp[:k]
    return arr


def call_tracker(tracker, style, detections, keypoints, frame):
    if style == "plain":
        return tracker.update(detections, frame)
    if style == "kps_first":
        return tracker.update(detections, keypoints, frame)
    return tracker.update(detections, frame, keypoints_to_array(keypoints, len(detections)))


# ============================================================
# INTERPOLATION / MOT RESULT EXPORT
# ============================================================

def interpolate_results(results, max_gap):
    by_id = defaultdict(list)
    for r in results:
        by_id[int(r[1])].append(np.asarray(r, dtype=np.float64))
    out = []
    for tid, rows in by_id.items():
        rows.sort(key=lambda r: r[0])
        out.append(rows[0])
        for a, b in zip(rows[:-1], rows[1:]):
            gap = int(b[0] - a[0])
            if 1 < gap <= max_gap:
                for k in range(1, gap):
                    r = a + (b - a) * (k / gap)
                    r[0], r[1] = a[0] + k, tid
                    out.append(r)
            out.append(b)
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def save_mot_txt(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},"
                f"{r[4]:.2f},{r[5]:.2f},{r[6]:.4f},-1,-1,-1\n"
            )


# ============================================================
# DEVICE
# ============================================================

def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def synchronize_cuda(device: str) -> None:
    if device and str(device).lower().startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="YOLO26n-Pose + BoT-SORT Ablation Runner")

    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    p.add_argument("--config-dir", type=str, default=str(CONFIG_DIR),
                   help="Thư mục chứa file config của từng thí nghiệm.")

    p.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT,
                   help="Tên file config (không đuôi .yaml); danh sách 'S1_torso,S2_thigh'; hoặc 'all'.")
    p.add_argument("--on", type=str, default="",
                   help=f"Bật thêm component (phân tách bằng dấu phẩy): {', '.join(COMPONENTS)}")
    p.add_argument("--off", type=str, default="",
                   help="Tắt component của config (phân tách bằng dấu phẩy).")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Ghi đè tham số (botsort / part / extractor), "
                        "vd: fusion_lambda=0.7 min_keypoint_conf=0.5")
    p.add_argument("--sweep", nargs="*", default=[], metavar="KEY=V1,V2",
                   help="Quét lưới tham số, vd: torso_shrink=0.6,0.8 feat_alpha=0.9,0.95")
    p.add_argument("--list", action="store_true", help="Liệt kê experiment/component/tham số rồi thoát.")

    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--device", type=str, default=get_default_device())
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)

    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES)
    p.add_argument("--no-latency-csv", action="store_true")
    p.add_argument("--no-results-txt", action="store_true")

    return p.parse_args()


# ============================================================
# BUILD TRACKER ARGUMENTS
# ============================================================

def build_tracker_args(args, run, overrides):
    cfg = run.cfg
    ta = SimpleNamespace()

    # --- BoT-SORT core ---
    for k, v in cfg["botsort"].items():
        setattr(ta, k, copy.deepcopy(v))

    ta.device = args.device
    ta.mot20 = run.mot20

    part_on = uses_part(run)

    # Original: with_reid = FastReID. Ours: with_reid = bật part-appearance.
    fr = cfg["fast_reid"]
    if run.original:
        ta.with_reid = bool(fr["enabled"])
        ta.fast_reid_config = fr["config"] if ta.with_reid else None
        ta.fast_reid_weights = fr["weights"] if ta.with_reid else None
    else:
        ta.with_reid = part_on
        ta.fast_reid_config = None
        ta.fast_reid_weights = None
    ta.with_kpt_appearance = part_on

    ta.fusion_mode = "weighted" if run.weighted_fusion else "min"

    # --- tham số + cờ thành phần của part-appearance (chỉ khi thật sự dùng) ---
    if part_on:
        for section in ("part", "extractor"):
            for k, v in cfg[section].items():
                setattr(ta, k, copy.deepcopy(v))
        for n in FLAG_COMPONENTS:
            setattr(ta, f"use_{n}", bool(getattr(run, n)))

    # --- ghi đè từ CLI ---
    for k, v in overrides.items():
        setattr(ta, k, v)

    return ta


def dump_run_config(ta, run, overrides, path: Path, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        tag=run.tag, experiment=run.name, config_file=str(run.config_path),
        tracker=run.cfg["tracker"],
        components={n: bool(getattr(run, n)) for n in COMPONENTS},
        overrides=overrides, tracker_args=vars(ta),
        video=str(args.video), model=str(args.model),
        imgsz=args.imgsz, conf=args.conf, iou=args.iou,
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

def _print_params(title, keys, ta, overrides):
    print("-" * 72)
    print(title)
    print("-" * 72)
    for k in keys:
        flag = "  *override" if k in overrides else ""
        print(f"  {k:<22}: {getattr(ta, k)}{flag}")


def print_runtime_configuration(run, ta, overrides):
    print()
    print("=" * 72)
    print(f"EXPERIMENT {run.tag}")
    print("=" * 72)
    print(f"Config file           : {run.config_path}")
    print(f"Description           : {run.description}")
    print(f"Tracker implementation: {'Original' if run.original else 'Experimental'} BoT-SORT")
    print(f"with_reid (effective) : {ta.with_reid}")
    print(f"mot20                 : {ta.mot20}")

    print("-" * 72)
    print("COMPONENTS")
    print("-" * 72)
    for title, names in COMPONENT_GROUPS:
        if title in ("EXTRACTOR", "TRACKER") and not uses_part(run):
            continue
        print(f"  [{title}]")
        for n in names:
            print(f"    {n:<18}: {'ON' if getattr(run, n) else 'off'}")

    if run.original and ta.with_reid:
        print("-" * 72)
        print(f"FastReID config  : {ta.fast_reid_config}")
        print(f"FastReID weights : {ta.fast_reid_weights}")

    # Chỉ in tham số thật sự có tác dụng với tổ hợp component hiện tại
    used = lambda keys: [k for k in keys if param_unused_reason(run, k) is None]
    _print_params("BOT-SORT PARAMS", used(YAML_KEYS), ta, overrides)

    if uses_part(run):
        print(f"  {'fusion_mode':<22}: {ta.fusion_mode}")
        _print_params("PART-APPEARANCE TRACKER PARAMS", used(PART_KEYS), ta, overrides)
        _print_params("EXTRACTOR PARAMS (-> PartAppearanceExtractor)", used(EXTRACTOR_KEYS), ta, overrides)

    print("=" * 72)
    print()


# ============================================================
# DETECTION CONVERSION
# ============================================================

def result_to_tracker_input(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32), []

    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    detections = np.concatenate([xyxy, conf[:, None]], axis=1).astype(np.float32)

    if result.keypoints is None:
        return detections, [None] * len(detections)

    kpt_xy = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
    if result.keypoints.conf is not None:
        kpt_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)
    else:
        kpt_conf = np.ones(kpt_xy.shape[:2], dtype=np.float32)

    keypoints = [np.concatenate([kpt_xy[i], kpt_conf[i][:, None]], axis=1)
                 for i in range(len(detections))]
    return detections, keypoints


# ============================================================
# VIDEO WRITER
# ============================================================

def create_video_writer(output_path, width, height, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer:\n{output_path}")
    return writer


# ============================================================
# LATENCY RECORDING / REPORT
# ============================================================

MODULE_STAGES = ("extract", "feature_update", "dist_stage1", "dist_unconfirmed")
COUNT_KEYS = ("n_extracted", "n_pairs", "n_pairs_unknown", "n_emb_better_than_iou",
              *[f"valid_part{p}" for p in range(NUM_PARTS)])


def make_latency_record(frame_id, n_det, yolo_ms, tracker_ms, tracker_latency):
    stage_ms = tracker_latency.get("ms", {}) if tracker_latency else {}
    counts = tracker_latency.get("counts", {}) if tracker_latency else {}

    record = {
        "frame": frame_id,
        "n_det": int(n_det),
        "yolo_ms": yolo_ms,
        "tracker_ms": tracker_ms,
    }
    for k in COUNT_KEYS:
        record[k] = int(counts.get(k, 0))

    for stage in MODULE_STAGES:
        record[f"{stage}_ms"] = float(stage_ms.get(stage, 0.0))

    record["module_ms"] = sum(record[f"{s}_ms"] for s in MODULE_STAGES)
    record["tracker_other_ms"] = max(0.0, tracker_ms - record["module_ms"])
    record["e2e_ms"] = yolo_ms + tracker_ms
    return record


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return dict(mean=0.0, median=0.0, p95=0.0, max=0.0, std=0.0)
    return dict(mean=float(arr.mean()), median=float(np.median(arr)),
                p95=float(np.percentile(arr, 95)), max=float(arr.max()), std=float(arr.std()))


def _print_row(name, values):
    s = _stats(values)
    print(f"{name:<36}{s['mean']:>9.3f}{s['median']:>9.3f}{s['p95']:>9.3f}{s['max']:>9.3f}{s['std']:>9.3f}")


def save_latency_csv(records, csv_path: Path):
    if not records:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def print_latency_report(records, warmup_frames, run, ta, device):
    if not records:
        print("No frames processed - no latency report.")
        return

    measured = records[warmup_frames:] if len(records) > warmup_frames else records
    used_warmup = warmup_frames if len(records) > warmup_frames else 0

    if run.original and ta.with_reid:
        module_name, module_enabled = "FastReID module (CNN)", True
    elif ta.with_kpt_appearance:
        module_name, module_enabled = "Part-appearance module", True
    else:
        module_name, module_enabled = "Appearance module (disabled)", False

    col = lambda key: [r[key] for r in measured]

    print()
    print("=" * 72)
    print("LATENCY REPORT (milliseconds)")
    print("=" * 72)
    print(f"Experiment      : {run.tag}")
    print(f"Appearance mod. : {module_name}")
    print(f"Device          : {device}")
    print(f"Frames total    : {len(records)} | warm-up skipped: {used_warmup} | measured: {len(measured)}")
    print()
    print(f"{'Stage':<36}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}{'std':>9}")
    print("-" * 81)

    _print_row("YOLO detect+pose (+ conversion)", col("yolo_ms"))
    _print_row("Tracker update (total)", col("tracker_ms"))

    if module_enabled:
        print(f"  [{module_name}]")
        for s in MODULE_STAGES:
            _print_row(f"    {s}", col(f"{s}_ms"))
        _print_row("    MODULE TOTAL", col("module_ms"))

    _print_row("  Tracker other (KF/GMC/IoU/assoc)", col("tracker_other_ms"))
    _print_row("END-TO-END (YOLO + tracker)", col("e2e_ms"))

    if module_enabled:
        mean_module = float(np.mean(col("module_ms")))
        mean_tracker = float(np.mean(col("tracker_ms")))
        mean_e2e = float(np.mean(col("e2e_ms")))
        total_extracted = int(np.sum(col("n_extracted")))
        total_extract_ms = float(np.sum(col("extract_ms")))
        total_module_ms = float(np.sum(col("module_ms")))

        print()
        print("-" * 72)
        print(f"{module_name} SUMMARY")
        print("-" * 72)
        print(f"Mean module latency / frame       : {mean_module:.3f} ms")
        if total_extracted > 0:
            print(f"Extraction latency / detection    : {total_extract_ms / total_extracted:.3f} ms")
            print(f"Module latency / detection        : {total_module_ms / total_extracted:.3f} ms")
            print(f"Avg. detections extracted / frame : {total_extracted / len(measured):.2f}")
            if ta.with_kpt_appearance:
                for p in range(NUM_PARTS):
                    if not getattr(ta, f"use_part_{PART_NAMES[p]}"):
                        print(f"Valid ratio part {p} ({PART_NAMES[p]:<5})       : disabled")
                        continue
                    v = int(np.sum(col(f"valid_part{p}")))
                    print(f"Valid ratio part {p} ({PART_NAMES[p]:<5})       : {100.0 * v / total_extracted:.1f} %")
        n_pairs = int(np.sum(col("n_pairs")))
        if n_pairs > 0:
            unk = int(np.sum(col("n_pairs_unknown")))
            print(f"Pairs without shared parts        : {100.0 * unk / n_pairs:.1f} %  (cost = 1.0 / IoU-only)")
            if ta.with_kpt_appearance and ta.fusion_mode == "min":
                better = int(np.sum(col("n_emb_better_than_iou")))
                print(f"Pairs where emb < iou (min mode)  : {100.0 * better / n_pairs:.1f} %")
        print(f"Share of tracker update           : {100.0 * mean_module / max(mean_tracker, 1e-9):.2f} %")
        print(f"Share of end-to-end frame latency : {100.0 * mean_module / max(mean_e2e, 1e-9):.2f} %")

    mean_e2e_all = float(np.mean(col("e2e_ms")))
    print()
    print("-" * 72)
    print(f"End-to-end mean latency : {mean_e2e_all:.3f} ms")
    print(f"End-to-end throughput   : {1000.0 / max(mean_e2e_all, 1e-9):.2f} FPS (visualisation excluded)")
    print("=" * 72)


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_one(args, run_spec, overrides, model):
    # tag có thêm hậu tố của override để các lần sweep không ghi đè lên nhau
    tag = run_spec.tag
    if overrides:
        tag += "_" + "_".join(f"{k}{v}" for k, v in overrides.items())
    run_spec = copy.copy(run_spec)
    run_spec.tag = tag

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video:\n{args.video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if video_fps <= 0:
        video_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker_args = build_tracker_args(args, run_spec, overrides)
    print_runtime_configuration(run_spec, tracker_args, overrides)
    print(f"Video resolution : {width} x {height}")
    print(f"Video frames     : {total_video_frames}")

    TrackerClass = get_tracker_class(run_spec)
    call_style = detect_call_style(TrackerClass)
    print(f"Tracker class    : {TrackerClass.__module__}.{TrackerClass.__name__} (update style: {call_style})")
    tracker = TrackerClass(tracker_args, frame_rate=video_fps)

    output_dir = Path(args.output_dir)
    video_stem = Path(args.video).stem
    output_path = output_dir / f"botsort_{tag}_{video_stem}.mp4"
    latency_csv_path = output_dir / f"latency_{tag}_{video_stem}.csv"
    results_path = output_dir / f"{tag}_{video_stem}.txt"
    config_path = output_dir / f"config_{tag}_{video_stem}.json"

    dump_run_config(tracker_args, run_spec, overrides, config_path, args)

    save_video = DEFAULT_SAVE_VIDEO and not args.no_save
    display_video = DEFAULT_DISPLAY and not args.no_display
    writer = create_video_writer(output_path, width, height, video_fps) if save_video else None
    if writer is not None:
        print(f"Output video    : {output_path}")

    frame_id = 0
    processed_frames = 0
    total_processing_time = 0.0
    latency_records = []
    mot_rows = []

    print(f"\nProcessing video ({tag})...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if args.max_frames is not None and frame_id > args.max_frames:
                break

            synchronize_cuda(args.device)
            t0 = time.perf_counter()

            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                                    iou=args.iou, device=args.device, verbose=False)
            detections, keypoints = result_to_tracker_input(results[0])

            synchronize_cuda(args.device)
            t1 = time.perf_counter()

            tracks = call_tracker(tracker, call_style, detections, keypoints, frame)

            synchronize_cuda(args.device)
            t2 = time.perf_counter()

            yolo_ms = (t1 - t0) * 1000.0
            tracker_ms = (t2 - t1) * 1000.0

            latency_records.append(make_latency_record(
                frame_id=frame_id, n_det=len(detections), yolo_ms=yolo_ms,
                tracker_ms=tracker_ms, tracker_latency=tracker.get_frame_latency()))

            for t in tracks:
                if not getattr(t, "is_activated", True):
                    continue
                x, y, w, h = [float(v) for v in t.tlwh]
                if w <= 0 or h <= 0:
                    continue
                mot_rows.append([frame_id, t.track_id, x, y, w, h, float(t.score)])

            elapsed = t2 - t0
            total_processing_time += elapsed
            processed_frames += 1

            current_fps = 1.0 / elapsed if elapsed > 1e-12 else 0.0
            average_fps = processed_frames / total_processing_time if total_processing_time > 1e-12 else 0.0

            if display_video or writer is not None:
                vis_frame = visualize_frame(
                    frame=frame.copy(), detections=detections, tracks=tracks,
                    frame_id=frame_id, total_frames=total_video_frames,
                    current_fps=current_fps, average_fps=average_fps)
                if writer is not None:
                    writer.write(vis_frame)
                if display_video:
                    cv2.imshow(f"YOLO26n-Pose + BoT-SORT [{tag}]", vis_frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        print("Stopped by user.")
                        break

            if processed_frames % 50 == 0:
                rec = latency_records[-1]
                print(f"[{tag}] Frame {processed_frames}/{total_video_frames} | "
                      f"FPS: {current_fps:.2f} | Avg FPS: {average_fps:.2f} | "
                      f"Det: {len(detections)} | Tracks: {len(tracks)} | "
                      f"Appearance: {rec['module_ms']:.2f} ms")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    final_fps = processed_frames / total_processing_time if total_processing_time > 1e-12 else 0.0
    print()
    print("=" * 72)
    print(f"PROCESSING COMPLETE [{tag}] | frames={processed_frames} | "
          f"time={total_processing_time:.3f}s | avg FPS={final_fps:.3f}")
    print("=" * 72)

    print_latency_report(latency_records, args.warmup_frames, run_spec, tracker_args, args.device)

    if not args.no_latency_csv and latency_records:
        save_latency_csv(latency_records, latency_csv_path)
        print(f"Per-frame latency CSV : {latency_csv_path}")

    if not args.no_results_txt and mot_rows:
        save_mot_txt(mot_rows, results_path)
        print(f"MOT results           : {results_path}")
        print(f"Run config            : {config_path}")

        if run_spec.interpolate:
            max_gap = int(getattr(tracker, "max_time_lost", 0) or getattr(tracker, "buffer_size", 30))
            rows_interp = interpolate_results(mot_rows, max_gap=max_gap)
            interp_path = results_path.with_name(results_path.stem + "_interp.txt")
            save_mot_txt(rows_interp, interp_path)
            print(f"MOT results (interp.) : {interp_path} "
                  f"(+{len(rows_interp) - len(mot_rows)} rows, max_gap={max_gap})")


# ============================================================
# MAIN
# ============================================================

def main():
    cli = parse_args()

    if cli.list:
        list_everything(cli.config_dir)
        return

    sel = cli.experiment.strip()
    if sel.lower() == "all":
        names = [p.stem for p in list_config_files(cli.config_dir).values()]
    else:
        names = _split(sel)
    on, off = _split(cli.on), _split(cli.off)
    base_overrides = parse_kv(cli.set)
    sweep = parse_sweep(cli.sweep)

    # Validate toàn bộ trước khi tải model / chạy
    jobs = []
    for name in names:
        spec = resolve_run(name, cli.config_dir, on, off)
        for combo in sweep:
            ov = {**base_overrides, **combo}
            check_overrides(spec, ov)
            jobs.append((spec, ov))

    print()
    print(f"Jobs ({len(jobs)}):")
    for spec, ov in jobs:
        print(f"  - {spec.tag}  {ov if ov else ''}")

    print("\nLoading YOLO model...")
    model = YOLO(cli.model)
    print("YOLO model loaded.")

    for spec, ov in jobs:
        print("\n" + "#" * 72)
        print(f"# RUN: {spec.tag} {ov if ov else ''}")
        print("#" * 72)
        run_one(copy.copy(cli), spec, ov, model)


if __name__ == "__main__":
    main()