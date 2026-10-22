"""
tracker/experiment_config.py
============================

Controlled ablation configuration for the BoT-SORT
keypoint-based appearance experiments.

Experiment table
----------------

| Group     | Exp | FastReID | KPT RGB | HSV | Conf.Gating | Conf.Weighting | Region | Multi-scale | BBox-norm | Adaptive Otsu+EMA |
| --------- | --- | -------- | ------- | --- | ----------- | -------------- | ------ | ----------- | --------- | ----------------- |
| Main      | E0  |    -     |    -    |  -  |      -      |       -        |   -    |      -      |     -     |         -         |
| Main      | E1  |    Y     |    -    |  -  |      -      |       -        |   -    |      -      |     -     |         -         |
| Main      | E2  |    -     |    Y    |  -  |      -      |       -        |   -    |      -      |     -     |         -         |
| Main      | E3  |    -     |    Y    |  -  |      Y      |       -        |   -    |      -      |     -     |         -         |
| Main      | E4  |    -     |    Y    |  -  |      Y      |       Y        |   -    |      -      |     -     |         -         |
| Main      | E5  |    -     |    Y    |  Y  |      Y      |       Y        |   -    |      -      |     -     |         -         |
| Extension | X1  |    -     |    Y    |  Y  |      Y      |       Y        |   Y    |      -      |     -     |         -         |
| Extension | X2  |    -     |    Y    |  Y  |      Y      |       Y        |   Y    |      Y      |     -     |         -         |
| Extension | X3  |    -     |    Y    |  Y  |      Y      |       Y        |   Y    |      Y      |     Y     |         -         |
| Extension | X4  |    -     |    Y    |  Y  |      Y      |       Y        |   Y    |      Y      |     Y     |         Y         |

(In the Main group, Region / Multi-scale / BBox-norm / Adaptive are all
OFF; "-" there means "not applicable", not a separate setting.)

Main group (E0 - E5)
--------------------

E0:
    Original BoT-SORT (IoU-only association, no appearance at all).

E1:
    E0 + FastReID appearance (conventional CNN ReID embedding).
    Reference point using a learned appearance model.
    (Independent branch: not part of the keypoint chain.)

E2:
    E0 + keypoint-guided local RGB patch appearance (no FastReID).
    No gating, no weighting, no HSV, no regions, single scale,
    fixed patch radius.

E3:
    E2 + keypoint validity gating by confidence
    (use_confidence_for_validity=True, fixed min_keypoint_confidence).

E4:
    E3 + continuous keypoint confidence weighting
    (enable_confidence_weighting=True).

E5:
    E4 + HSV color channel (enable_hsv_appearance=True).
    End of the Main group: RGB + HSV + gating + weighting.

Extension group (X1 - X4), built on top of E5
---------------------------------------------

X1:
    E5 + anatomical region aggregation
    (enable_region_appearance=True: head / torso / legs).

X2:
    X1 + multi-scale patch extraction
    (enable_multiscale_patch=True).

X3:
    X2 + bounding-box-normalized patch radius
    (enable_bbox_normalized_patch=True).

X4:
    X3 + adaptive (Otsu) confidence threshold with temporal EMA
    smoothing (enable_adaptive_confidence_threshold=True).
    Full model.

Each stage is a strict superset of the previous one in its chain
(E2 -> E3 -> E4 -> E5 -> X1 -> X2 -> X3 -> X4). This is enforced by
building every stage with dataclasses.replace() from its predecessor,
so only the newly introduced switch(es) are written down.

Association structure (unchanged across all stages)
-----------------------------------------------------

First association:
    IoU
    + detection-score fusion
    + custom appearance (from E2 onward)
    + proximity gating
    + minimum fusion
    + Hungarian assignment

Second association:
    IoU only

Unconfirmed association:
    IoU
    + detection-score fusion
    + custom appearance (from E2 onward)
    + proximity gating
    + minimum fusion
    + Hungarian assignment

FastReID
--------

E1 only. E2-E5 and X1-X4 use custom keypoint-based appearance only.

Memory / visibility-aware extensions
-------------------------------------

Temporal region memory, missing-region tolerance, region
visibility weighting, and a visibility-aware appearance distance
are NOT part of this ablation. The compatibility fields are kept on
ExperimentConfig (disabled / neutral) purely so existing tooling
(e.g. print_experiment) does not break.
"""


from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


FAST_REID_CONFIG = str(
    PROJECT_ROOT
    / "fast_reid"
    / "configs"
    / "MOT20"
    / "sbs_S50.yml"
)


FAST_REID_WEIGHTS = str(
    PROJECT_ROOT
    / "fast_reid"
    / "weights"
    / "mot20_sbs_S50.pth"
)


# ============================================================
# Experiment configuration
# ============================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for one controlled BoT-SORT experiment.

    This class only defines configuration.

    It does NOT:
        - load YOLO
        - load FastReID
        - load model weights
        - perform tracking
        - perform association
        - extract appearance features
    """

    # ========================================================
    # Basic experiment information
    # ========================================================

    name: str

    description: str

    # ========================================================
    # BoT-SORT compatibility
    # ========================================================

    mot20: bool = False

    # ========================================================
    # FastReID
    # ========================================================

    with_reid: bool = False

    fast_reid_config: str = ""

    fast_reid_weights: str = ""

    # ========================================================
    # Custom keypoint appearance
    # ========================================================

    with_kpt_appearance: bool = False

    # ========================================================
    # Appearance switches
    # ========================================================

    enable_region_appearance: bool = False

    enable_confidence_weighting: bool = False

    enable_bbox_normalized_patch: bool = False

    enable_hsv_appearance: bool = False

    enable_multiscale_patch: bool = False

    # ========================================================
    # Keypoint patch parameters
    # ========================================================

    patch_radius: int = 8

    patch_resize: int = 16

    # Canonical extractor parameter.
    #
    # KeypointPatchAppearanceExtractor receives:
    #     resize_size=args.resize_size
    #
    resize_size: int = 16

    patch_radius_ratio: float = 0.02

    keypoint_indices: Optional[List[int]] = None

    min_keypoint_confidence: float = 0.3

    # ========================================================
    # Confidence weighting / gating parameters
    # ========================================================

    confidence_power: float = 1.0

    use_confidence_for_validity: bool = False

    # ========================================================
    # Adaptive (Otsu) confidence threshold parameters
    # ========================================================

    enable_adaptive_confidence_threshold: bool = False

    # Floor guard: adaptive threshold never drops below
    # min_keypoint_confidence * adaptive_floor_ratio.
    adaptive_floor_ratio: float = 0.3

    # Histogram resolution used by Otsu's method.
    adaptive_otsu_bins: int = 64

    # Minimum number of confidence samples required in a single
    # extract() call before the Otsu estimate is trusted; below
    # this, falls back to the fixed min_keypoint_confidence for
    # that call.
    adaptive_min_samples: int = 30

    # EMA smoothing factor for the adaptive threshold across
    # successive extract() calls (i.e. across frames).
    # Closer to 1.0 = smoother / slower to react.
    adaptive_ema_alpha: float = 0.9

    # ========================================================
    # Multi-scale appearance
    # ========================================================

    patch_scales: Tuple[float, ...] = (0.5, 1.0, 2.0)

    scale_weights: Tuple[float, ...] = (0.25, 0.50, 0.25)

    # --------------------------------------------------------
    # Backward-compatible aliases
    # --------------------------------------------------------

    multiscale_scales: Tuple[float, ...] = (0.5, 1.0, 2.0)

    multiscale_weights: Tuple[float, ...] = (0.25, 0.50, 0.25)

    # ========================================================
    # Anatomical regions
    # ========================================================

    region_indices: Optional[Dict[str, List[int]]] = None

    auto_normalize_coordinates: bool = True

    # ========================================================
    # Memory / visibility-aware extensions
    #
    # NOT part of the ablation. Kept disabled / neutral so
    # existing interfaces (e.g. print_experiment) keep working.
    # ========================================================

    enable_visibility_aware_distance: bool = False

    appearance_distance_metric: str = "cosine"

    appearance_missing_distance: float = 1.0

    min_joint_valid: int = 0


# ============================================================
# Common defaults
# ============================================================

_DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.3


# ============================================================
# MAIN GROUP
# ============================================================

# ------------------------------------------------------------
# E0 - Original BoT-SORT (IoU only, no appearance)
# ------------------------------------------------------------

E0 = ExperimentConfig(
    name="E0",
    description=(
        "Baseline: IoU-only BoT-SORT association. "
        "No appearance model of any kind."
    ),
    mot20=False,
    with_reid=False,
    with_kpt_appearance=False,
    min_keypoint_confidence=0.0,
)


# ------------------------------------------------------------
# E1 - E0 + FastReID (conventional CNN ReID appearance)
# ------------------------------------------------------------

E1 = replace(
    E0,
    name="E1",
    description=(
        "E0 + FastReID appearance embedding. "
        "Conventional learned ReID reference point."
    ),
    with_reid=True,
    fast_reid_config=FAST_REID_CONFIG,
    fast_reid_weights=FAST_REID_WEIGHTS,
)


# ------------------------------------------------------------
# E2 - E0 + keypoint-guided local RGB patch appearance
# ------------------------------------------------------------

E2 = replace(
    E0,
    name="E2",
    description=(
        "E0 + keypoint-guided local RGB patch appearance. "
        "No FastReID. No gating, weighting, HSV, regions, "
        "multi-scale or bbox normalization; fixed patch radius."
    ),
    with_kpt_appearance=True,
)


# ------------------------------------------------------------
# E3 - E2 + confidence-based keypoint validity gating
# ------------------------------------------------------------

E3 = replace(
    E2,
    name="E3",
    description=(
        "E2 + keypoint validity gating by confidence "
        "(fixed min_keypoint_confidence threshold). Low-"
        "confidence / occluded keypoints are dropped entirely."
    ),
    min_keypoint_confidence=_DEFAULT_MIN_KEYPOINT_CONFIDENCE,
    use_confidence_for_validity=True,
)


# ------------------------------------------------------------
# E4 - E3 + continuous keypoint confidence weighting
# ------------------------------------------------------------

E4 = replace(
    E3,
    name="E4",
    description=(
        "E3 + continuous keypoint confidence weighting. Valid "
        "keypoints contribute proportionally to their "
        "confidence instead of equally."
    ),
    enable_confidence_weighting=True,
)


# ------------------------------------------------------------
# E5 - E4 + HSV color channel  (end of Main group)
# ------------------------------------------------------------

E5 = replace(
    E4,
    name="E5",
    description=(
        "E4 + HSV color channel (RGB + HSV). Adds "
        "illumination-invariant color information to the "
        "keypoint patch descriptor. End of the Main group."
    ),
    enable_hsv_appearance=True,
)


# ============================================================
# EXTENSION GROUP (built on top of E5)
# ============================================================

# ------------------------------------------------------------
# X1 - E5 + anatomical region aggregation
# ------------------------------------------------------------

X1 = replace(
    E5,
    name="X1",
    description=(
        "E5 + head / torso / legs region aggregation. Tests "
        "whether structuring the descriptor by anatomical "
        "region adds robustness over a flat keypoint concat."
    ),
    enable_region_appearance=True,
)


# ------------------------------------------------------------
# X2 - X1 + multi-scale patch extraction
# ------------------------------------------------------------

X2 = replace(
    X1,
    name="X2",
    description=(
        "X1 + multi-scale patch extraction. Adds "
        "scale-robust local appearance."
    ),
    enable_multiscale_patch=True,
)


# ------------------------------------------------------------
# X3 - X2 + bounding-box-normalized patch radius
# ------------------------------------------------------------

X3 = replace(
    X2,
    name="X3",
    description=(
        "X2 + bounding-box-normalized patch radius. Patch "
        "size scales with the pedestrian's own bbox height "
        "instead of a fixed pixel radius."
    ),
    enable_bbox_normalized_patch=True,
)


# ------------------------------------------------------------
# X4 - X3 + adaptive (Otsu) confidence threshold + EMA
#      Final, full model.
# ------------------------------------------------------------

X4 = replace(
    X3,
    name="X4",
    description=(
        "X3 + adaptive (Otsu) confidence threshold, smoothed "
        "across frames with an EMA, replacing the fixed "
        "min_keypoint_confidence cutoff. Full model."
    ),
    # min_keypoint_confidence is kept: it is the fallback when
    # the adaptive estimate is not trusted (too few samples) and
    # the basis of the floor guard.
    enable_adaptive_confidence_threshold=True,
)


# ============================================================
# Experiment registry
# ============================================================

EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "E0": E0,
    "E1": E1,
    "E2": E2,
    "E3": E3,
    "E4": E4,
    "E5": E5,
    "X1": X1,
    "X2": X2,
    "X3": X3,
    "X4": X4,
}

MAIN_EXPERIMENTS: List[str] = ["E0", "E1", "E2", "E3", "E4", "E5"]

EXTENSION_EXPERIMENTS: List[str] = ["X1", "X2", "X3", "X4"]


# ============================================================
# Get experiment
# ============================================================

def get_experiment(
    name: str,
) -> ExperimentConfig:
    """
    Get experiment configuration by name.

    Parameters
    ----------
    name : str
        E0, ..., E5, X1, ..., X4.

    Returns
    -------
    ExperimentConfig
        Selected experiment configuration.
    """

    name = str(name).upper().strip()

    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{name}'. "
            f"Available experiments: "
            f"{list(EXPERIMENTS.keys())}"
        )

    return EXPERIMENTS[name]


# ============================================================
# Print experiment
# ============================================================

def print_experiment(
    config: ExperimentConfig,
) -> None:
    """
    Print complete experiment configuration.
    """

    def _row(label: str, attr: str) -> None:
        print(f"{label:<30}= {getattr(config, attr)}")

    print()
    print("=" * 78)
    print(f"Experiment: {config.name}")
    print(f"Description: {config.description}")
    print("-" * 78)

    # Basic
    _row("mot20", "mot20")
    _row("with_reid", "with_reid")
    _row("with_kpt_appearance", "with_kpt_appearance")

    # Appearance switches
    _row("enable_region_appearance", "enable_region_appearance")
    _row("enable_confidence_weighting", "enable_confidence_weighting")
    _row("enable_bbox_normalized_patch", "enable_bbox_normalized_patch")
    _row("enable_hsv_appearance", "enable_hsv_appearance")
    _row("enable_multiscale_patch", "enable_multiscale_patch")
    print("-" * 78)

    # Patch parameters
    _row("patch_radius", "patch_radius")
    _row("patch_resize", "patch_resize")
    _row("resize_size", "resize_size")
    _row("patch_radius_ratio", "patch_radius_ratio")
    _row("keypoint_indices", "keypoint_indices")
    _row("min_keypoint_confidence", "min_keypoint_confidence")

    # Confidence parameters
    _row("confidence_power", "confidence_power")
    _row("use_confidence_for_validity", "use_confidence_for_validity")

    # Adaptive (Otsu)
    _row("enable_adaptive_conf_thresh",
         "enable_adaptive_confidence_threshold")
    _row("adaptive_floor_ratio", "adaptive_floor_ratio")
    _row("adaptive_otsu_bins", "adaptive_otsu_bins")
    _row("adaptive_min_samples", "adaptive_min_samples")
    _row("adaptive_ema_alpha", "adaptive_ema_alpha")

    # Multi-scale
    _row("patch_scales", "patch_scales")
    _row("scale_weights", "scale_weights")
    _row("multiscale_scales", "multiscale_scales")
    _row("multiscale_weights", "multiscale_weights")

    # Regions
    _row("region_indices", "region_indices")
    _row("auto_normalize_coordinates", "auto_normalize_coordinates")
    print("-" * 78)

    # Appearance distance (compatibility only, unused)
    _row("enable_visibility_aware_dist.",
         "enable_visibility_aware_distance")
    _row("appearance_distance_metric", "appearance_distance_metric")
    _row("appearance_missing_distance", "appearance_missing_distance")
    _row("min_joint_valid", "min_joint_valid")
    print("-" * 78)

    # FastReID
    _row("fast_reid_config", "fast_reid_config")
    _row("fast_reid_weights", "fast_reid_weights")

    print("=" * 78)
    print()