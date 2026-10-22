"""
tracker/appearance_distance.py
==============================

Visibility-aware appearance distance for BoT-SORT.

Responsibilities
----------------

This module compares appearance descriptors.

Implemented:

    1. Cosine distance
    2. Visibility-aware comparison
    3. Keypoint-level valid masks
    4. Region-level valid masks
    5. Confidence weighting
    6. Missing-region tolerance
    7. Minimum common visibility gating

NOT implemented:

    - feature extraction
    - temporal memory
    - Kalman
    - GMC
    - track management


Core idea
---------

Instead of:

    cosine(track_feature, detection_feature)

use only jointly visible parts:

    D =
        sum_i w_i * d_i
        ----------------
        sum_i w_i


where:

    w_i =
        track_valid_i
        *
        detection_valid_i
        *
        optional confidence weights


This prevents missing regions from behaving
as if they were real appearance information.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# Utilities
# ============================================================================


def _safe_float(
    value,
    default: float = 0.0,
) -> float:

    try:

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):

            return float(
                default
            )

        return value

    except (
        TypeError,
        ValueError,
    ):

        return float(
            default
        )


def _l2_normalize(
    feature: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:

    feature = np.asarray(
        feature,
        dtype=np.float32,
    )

    norm = float(
        np.linalg.norm(
            feature
        )
    )

    if norm <= eps:

        return np.zeros_like(
            feature,
            dtype=np.float32,
        )

    return (
        feature / norm
    ).astype(
        np.float32
    )


def cosine_distance(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Standard cosine distance.

    d = 1 - cosine_similarity
    """

    a = np.asarray(
        feature_a,
        dtype=np.float32,
    )

    b = np.asarray(
        feature_b,
        dtype=np.float32,
    )

    if a.shape != b.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"{a.shape} vs {b.shape}"
        )

    norm_a = float(
        np.linalg.norm(a)
    )

    norm_b = float(
        np.linalg.norm(b)
    )

    if (
        norm_a <= eps
        or norm_b <= eps
    ):

        return 1.0

    similarity = float(
        np.dot(a, b)
        /
        (
            norm_a
            * norm_b
        )
    )

    similarity = float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )

    return float(
        1.0 - similarity
    )


# ============================================================================
# Main class
# ============================================================================


class AppearanceDistance:
    """
    Appearance distance calculator.

    Parameters
    ----------

    enable_visibility_aware_distance:
        Main switch.

        False:
            ordinary cosine distance.

        True:
            compare only jointly valid slots.

    min_common_visibility:
        Minimum fraction of jointly visible slots.

    missing_region_penalty:
        Optional penalty when valid appearance
        overlap is insufficient.

        Recommended default:

            0.0

        because association layer can perform
        the final gating.

    enable_confidence_weighting:
        Use confidence as distance weights.

    confidence_power:
        Confidence exponent.

    normalize_region_weights:
        Normalize weights so their sum is 1.

    verbose:
        Print configuration.
    """

    def __init__(
        self,

        enable_visibility_aware_distance: bool = False,

        min_common_visibility: float = 0.0,

        missing_region_penalty: float = 0.0,

        enable_confidence_weighting: bool = False,

        confidence_power: float = 1.0,

        normalize_region_weights: bool = True,

        verbose: bool = False,
    ):

        self.enable_visibility_aware_distance = bool(
            enable_visibility_aware_distance
        )

        self.min_common_visibility = float(
            min_common_visibility
        )

        self.missing_region_penalty = float(
            missing_region_penalty
        )

        self.enable_confidence_weighting = bool(
            enable_confidence_weighting
        )

        self.confidence_power = float(
            confidence_power
        )

        self.normalize_region_weights = bool(
            normalize_region_weights
        )

        self.verbose = bool(
            verbose
        )

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

        if not (
            0.0
            <= self.min_common_visibility
            <= 1.0
        ):

            raise ValueError(
                "min_common_visibility "
                "must be in [0,1]"
            )

        if self.missing_region_penalty < 0:

            raise ValueError(
                "missing_region_penalty "
                "must be >= 0"
            )

        if self.confidence_power <= 0:

            raise ValueError(
                "confidence_power "
                "must be > 0"
            )

        if self.verbose:

            self.print_config()

    # ======================================================================
    # Configuration
    # ======================================================================

    def get_config(self) -> Dict:

        return {
            "enable_visibility_aware_distance":
                self.enable_visibility_aware_distance,

            "min_common_visibility":
                self.min_common_visibility,

            "missing_region_penalty":
                self.missing_region_penalty,

            "enable_confidence_weighting":
                self.enable_confidence_weighting,

            "confidence_power":
                self.confidence_power,

            "normalize_region_weights":
                self.normalize_region_weights,
        }

    def print_config(self):

        print(
            "\n"
            "============================================================\n"
            "AppearanceDistance\n"
            "============================================================\n"
            f"visibility_aware_distance : "
            f"{self.enable_visibility_aware_distance}\n"

            f"min_common_visibility     : "
            f"{self.min_common_visibility}\n"

            f"missing_region_penalty   : "
            f"{self.missing_region_penalty}\n"

            f"confidence_weighting     : "
            f"{self.enable_confidence_weighting}\n"

            f"confidence_power         : "
            f"{self.confidence_power}\n"

            "============================================================\n"
        )

    # ======================================================================
    # Flat mask
    # ======================================================================

    def _prepare_mask(
        self,
        mask,
        feature_dim: int,
        descriptor_dim_per_slot: int,
    ) -> np.ndarray:
        """
        Convert slot-level mask into feature-level mask.

        Example:

            valid_mask =
                [True, False, True]

        descriptor_dim_per_slot = 12

        becomes:

            [True x 12,
             False x 12,
             True x 12]
        """

        if mask is None:

            return np.ones(
                feature_dim,
                dtype=bool,
            )

        mask = np.asarray(
            mask,
            dtype=bool,
        ).reshape(-1)

        if mask.size == feature_dim:

            return mask

        if (
            descriptor_dim_per_slot > 0
            and (
                feature_dim
                %
                descriptor_dim_per_slot
                == 0
            )
        ):

            expected_slots = (
                feature_dim
                //
                descriptor_dim_per_slot
            )

            if mask.size == expected_slots:

                return np.repeat(
                    mask,
                    descriptor_dim_per_slot,
                )

        raise ValueError(
            "Mask dimension incompatible "
            f"with feature dimension. "
            f"mask={mask.size}, "
            f"feature={feature_dim}, "
            f"slot_dim={descriptor_dim_per_slot}"
        )

    # ======================================================================
    # Slot distance
    # ======================================================================

    def slotwise_cosine_distance(
        self,
        feature_a: np.ndarray,
        feature_b: np.ndarray,
        valid_mask_a: Optional[np.ndarray] = None,
        valid_mask_b: Optional[np.ndarray] = None,
        slot_confidence_a: Optional[np.ndarray] = None,
        slot_confidence_b: Optional[np.ndarray] = None,
        descriptor_dim_per_slot: int = 1,
    ) -> Tuple[float, Dict]:
        """
        Visibility-aware slot-level cosine distance.

        Each slot is compared independently.

        Example:

            17 keypoints
            12-D per keypoint

        descriptor_dim_per_slot = 12
        """

        a = np.asarray(
            feature_a,
            dtype=np.float32,
        ).reshape(-1)

        b = np.asarray(
            feature_b,
            dtype=np.float32,
        ).reshape(-1)

        if a.shape != b.shape:

            raise ValueError(
                f"Feature shape mismatch: "
                f"{a.shape} vs {b.shape}"
            )

        feature_dim = a.size

        if descriptor_dim_per_slot <= 0:

            raise ValueError(
                "descriptor_dim_per_slot "
                "must be > 0"
            )

        if (
            feature_dim
            %
            descriptor_dim_per_slot
            != 0
        ):

            raise ValueError(
                "feature_dim must be divisible "
                "by descriptor_dim_per_slot"
            )

        num_slots = (
            feature_dim
            //
            descriptor_dim_per_slot
        )

        mask_a = self._prepare_mask(
            valid_mask_a,
            feature_dim,
            descriptor_dim_per_slot,
        )

        mask_b = self._prepare_mask(
            valid_mask_b,
            feature_dim,
            descriptor_dim_per_slot,
        )

        mask_a = mask_a.reshape(
            num_slots,
            descriptor_dim_per_slot,
        )

        mask_b = mask_b.reshape(
            num_slots,
            descriptor_dim_per_slot,
        )

        slot_valid_a = np.any(
            mask_a,
            axis=1,
        )

        slot_valid_b = np.any(
            mask_b,
            axis=1,
        )

        common_mask = (
            slot_valid_a
            &
            slot_valid_b
        )

        common_count = int(
            np.sum(
                common_mask
            )
        )

        common_visibility = (
            float(common_count)
            /
            float(num_slots)
            if num_slots > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # No common appearance
        # ------------------------------------------------------------------

        if common_count == 0:

            return (
                float(
                    self.missing_region_penalty
                ),
                {
                    "distance":
                        float(
                            self.missing_region_penalty
                        ),

                    "num_slots":
                        num_slots,

                    "common_slots":
                        0,

                    "common_visibility":
                        0.0,

                    "gated":
                        True,

                    "slot_distances":
                        np.full(
                            num_slots,
                            np.nan,
                            dtype=np.float32,
                        ),
                },
            )

        # ------------------------------------------------------------------
        # Visibility gate
        # ------------------------------------------------------------------

        if (
            self.enable_visibility_aware_distance
            and common_visibility
            < self.min_common_visibility
        ):

            return (
                float(
                    self.missing_region_penalty
                ),
                {
                    "distance":
                        float(
                            self.missing_region_penalty
                        ),

                    "num_slots":
                        num_slots,

                    "common_slots":
                        common_count,

                    "common_visibility":
                        common_visibility,

                    "gated":
                        True,

                    "slot_distances":
                        np.full(
                            num_slots,
                            np.nan,
                            dtype=np.float32,
                        ),
                },
            )

        # ------------------------------------------------------------------
        # Confidence
        # ------------------------------------------------------------------

        conf_a = None
        conf_b = None

        if (
            self.enable_confidence_weighting
        ):

            if slot_confidence_a is not None:

                conf_a = np.asarray(
                    slot_confidence_a,
                    dtype=np.float32,
                ).reshape(-1)

                if conf_a.size != num_slots:

                    raise ValueError(
                        "slot_confidence_a "
                        "dimension mismatch"
                    )

            if slot_confidence_b is not None:

                conf_b = np.asarray(
                    slot_confidence_b,
                    dtype=np.float32,
                ).reshape(-1)

                if conf_b.size != num_slots:

                    raise ValueError(
                        "slot_confidence_b "
                        "dimension mismatch"
                    )

        # ------------------------------------------------------------------
        # Slot comparison
        # ------------------------------------------------------------------

        slot_distances = np.full(
            num_slots,
            np.nan,
            dtype=np.float32,
        )

        weights = np.zeros(
            num_slots,
            dtype=np.float32,
        )

        for slot_idx in range(
            num_slots
        ):

            if not common_mask[
                slot_idx
            ]:

                continue

            start = (
                slot_idx
                *
                descriptor_dim_per_slot
            )

            end = (
                start
                +
                descriptor_dim_per_slot
            )

            fa = a[
                start:end
            ]

            fb = b[
                start:end
            ]

            distance = cosine_distance(
                fa,
                fb,
            )

            slot_distances[
                slot_idx
            ] = float(
                distance
            )

            weight = 1.0

            if (
                self.enable_confidence_weighting
            ):

                ca = (
                    1.0
                    if conf_a is None
                    else float(
                        np.clip(
                            conf_a[
                                slot_idx
                            ],
                            0.0,
                            1.0,
                        )
                    )
                )

                cb = (
                    1.0
                    if conf_b is None
                    else float(
                        np.clip(
                            conf_b[
                                slot_idx
                            ],
                            0.0,
                            1.0,
                        )
                    )
                )

                weight = (
                    (
                        ca
                        *
                        cb
                    )
                    **
                    self.confidence_power
                )

            weights[
                slot_idx
            ] = float(
                weight
            )

        # ------------------------------------------------------------------
        # Weighted mean
        # ------------------------------------------------------------------

        active = (
            weights > 0
        ) & common_mask

        if not np.any(
            active
        ):

            return (
                float(
                    self.missing_region_penalty
                ),
                {
                    "distance":
                        float(
                            self.missing_region_penalty
                        ),

                    "num_slots":
                        num_slots,

                    "common_slots":
                        common_count,

                    "common_visibility":
                        common_visibility,

                    "gated":
                        True,

                    "slot_distances":
                        slot_distances,
                },
            )

        active_distances = (
            slot_distances[
                active
            ]
        )

        active_weights = (
            weights[
                active
            ]
        )

        if self.normalize_region_weights:

            denominator = float(
                np.sum(
                    active_weights
                )
            )

            if denominator <= 1e-12:

                distance = float(
                    np.mean(
                        active_distances
                    )
                )

            else:

                distance = float(
                    np.sum(
                        active_distances
                        *
                        active_weights
                    )
                    /
                    denominator
                )

        else:

            distance = float(
                np.sum(
                    active_distances
                    *
                    active_weights
                )
            )

        return (
            distance,
            {
                "distance":
                    distance,

                "num_slots":
                    num_slots,

                "common_slots":
                    common_count,

                "common_visibility":
                    common_visibility,

                "gated":
                    False,

                "slot_distances":
                    slot_distances,

                "slot_weights":
                    weights,
            },
        )

    # ======================================================================
    # Region distance
    # ======================================================================

    def region_distance(
        self,
        track_regions: Dict[str, np.ndarray],
        detection_regions: Dict[str, np.ndarray],
        track_metadata: Optional[
            Dict[str, Dict]
        ] = None,
        detection_metadata: Optional[
            Dict[str, Dict]
        ] = None,
        region_weights: Optional[
            Dict[str, float]
        ] = None,
        descriptor_dim_per_slot: int = 1,
    ) -> Tuple[float, Dict]:
        """
        Calculate region-level appearance distance.

        Regions are compared independently:

            head
            torso
            legs

        Missing regions are ignored.
        """

        if region_weights is None:

            region_weights = {
                "head": 1.0,
                "torso": 1.0,
                "legs": 1.0,
            }

        distances = {}
        weights = {}

        for region_name in (
            "head",
            "torso",
            "legs",
        ):

            if (
                region_name
                not in track_regions
                or region_name
                not in detection_regions
            ):

                continue

            feature_a = (
                track_regions[
                    region_name
                ]
            )

            feature_b = (
                detection_regions[
                    region_name
                ]
            )

            valid_a = None
            valid_b = None

            conf_a = None
            conf_b = None

            if track_metadata is not None:

                meta = track_metadata.get(
                    region_name,
                    {},
                )

                valid_a = meta.get(
                    "valid_mask"
                )

                conf_a = meta.get(
                    "keypoint_confidences"
                )

            if detection_metadata is not None:

                meta = detection_metadata.get(
                    region_name,
                    {},
                )

                valid_b = meta.get(
                    "valid_mask"
                )

                conf_b = meta.get(
                    "keypoint_confidences"
                )

            distance, info = (
                self.slotwise_cosine_distance(
                    feature_a=feature_a,
                    feature_b=feature_b,
                    valid_mask_a=valid_a,
                    valid_mask_b=valid_b,
                    slot_confidence_a=conf_a,
                    slot_confidence_b=conf_b,
                    descriptor_dim_per_slot=(
                        descriptor_dim_per_slot
                    ),
                )
            )

            if info["gated"]:

                continue

            distances[
                region_name
            ] = distance

            weights[
                region_name
            ] = float(
                region_weights.get(
                    region_name,
                    1.0,
                )
            )

        # ------------------------------------------------------------------
        # No region available
        # ------------------------------------------------------------------

        if not distances:

            return (
                float(
                    self.missing_region_penalty
                ),
                {
                    "distance":
                        float(
                            self.missing_region_penalty
                        ),

                    "region_distances":
                        {},

                    "region_weights":
                        {},

                    "common_regions":
                        0,

                    "gated":
                        True,
                },
            )

        # ------------------------------------------------------------------
        # Weighted region distance
        # ------------------------------------------------------------------

        region_names = list(
            distances.keys()
        )

        numerator = 0.0
        denominator = 0.0

        for region_name in region_names:

            weight = float(
                weights[
                    region_name
                ]
            )

            numerator += (
                weight
                *
                distances[
                    region_name
                ]
            )

            denominator += weight

        if denominator <= 1e-12:

            final_distance = float(
                np.mean(
                    list(
                        distances.values()
                    )
                )
            )

        else:

            final_distance = (
                numerator
                /
                denominator
            )

        return (
            float(
                final_distance
            ),
            {
                "distance":
                    float(
                        final_distance
                    ),

                "region_distances":
                    distances,

                "region_weights":
                    weights,

                "common_regions":
                    len(
                        distances
                    ),

                "gated":
                    False,
            },
        )

    # ======================================================================
    # Generic API
    # ======================================================================

    def distance(
        self,
        feature_a: np.ndarray,
        feature_b: np.ndarray,
        valid_mask_a: Optional[np.ndarray] = None,
        valid_mask_b: Optional[np.ndarray] = None,
        descriptor_dim_per_slot: int = 1,
        confidence_a: Optional[np.ndarray] = None,
        confidence_b: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """
        Generic distance API.
        """

        if (
            not self.enable_visibility_aware_distance
        ):

            distance = cosine_distance(
                feature_a,
                feature_b,
            )

            return (
                distance,
                {
                    "distance":
                        distance,

                    "gated":
                        False,

                    "common_visibility":
                        1.0,

                    "common_slots":
                        None,
                },
            )

        return (
            self.slotwise_cosine_distance(
                feature_a=feature_a,
                feature_b=feature_b,
                valid_mask_a=valid_mask_a,
                valid_mask_b=valid_mask_b,
                slot_confidence_a=confidence_a,
                slot_confidence_b=confidence_b,
                descriptor_dim_per_slot=(
                    descriptor_dim_per_slot
                ),
            )
        )


# ============================================================================
# Factory
# ============================================================================


def create_appearance_distance(
    enable_visibility_aware_distance: bool = True,
    min_common_visibility: float = 0.25,
    missing_region_penalty: float = 1.0,
    enable_confidence_weighting: bool = False,
    confidence_power: float = 1.0,
):
    """
    Recommended factory.
    """

    return AppearanceDistance(
        enable_visibility_aware_distance=(
            enable_visibility_aware_distance
        ),

        min_common_visibility=(
            min_common_visibility
        ),

        missing_region_penalty=(
            missing_region_penalty
        ),

        enable_confidence_weighting=(
            enable_confidence_weighting
        ),

        confidence_power=(
            confidence_power
        ),
    )


# ============================================================================
# Self-test
# ============================================================================


if __name__ == "__main__":

    print(
        "\n"
        "============================================================\n"
        "APPEARANCE DISTANCE SELF TEST\n"
        "============================================================"
    )

    rng = np.random.default_rng(
        42
    )

    # ------------------------------------------------------------------
    # 4 slots x 6 dimensions
    # ------------------------------------------------------------------

    a = rng.random(
        24
    ).astype(
        np.float32
    )

    b = rng.random(
        24
    ).astype(
        np.float32
    )

    mask_a = np.array(
        [
            True,
            True,
            False,
            True,
        ],
        dtype=bool,
    )

    mask_b = np.array(
        [
            True,
            False,
            True,
            True,
        ],
        dtype=bool,
    )

    # ------------------------------------------------------------------
    # Standard
    # ------------------------------------------------------------------

    standard = AppearanceDistance(
        enable_visibility_aware_distance=False
    )

    d_standard, info_standard = (
        standard.distance(
            a,
            b,
        )
    )

    print(
        "\nStandard distance:",
        d_standard,
    )

    # ------------------------------------------------------------------
    # Visibility aware
    # ------------------------------------------------------------------

    visible = AppearanceDistance(
        enable_visibility_aware_distance=True,
        min_common_visibility=0.25,
    )

    d_visible, info_visible = (
        visible.distance(
            a,
            b,
            valid_mask_a=mask_a,
            valid_mask_b=mask_b,
            descriptor_dim_per_slot=6,
        )
    )

    print(
        "\nVisibility-aware distance:",
        d_visible,
    )

    print(
        "Common slots:",
        info_visible[
            "common_slots"
        ],
    )

    print(
        "Common visibility:",
        info_visible[
            "common_visibility"
        ],
    )

    assert (
        info_visible[
            "common_slots"
        ]
        == 2
    )

    print(
        "\n"
        "============================================================\n"
        "SELF TEST PASSED\n"
        "============================================================"
    )