"""
tracker/appearance_memory.py
============================

Temporal region appearance memory for BoT-SORT.

Responsibilities
----------------

This module ONLY manages temporal appearance memory.

Implemented:

    1. Head / torso / legs memory
    2. EMA update
    3. Visibility-aware update
    4. Missing-region freeze
    5. Memory aging
    6. Per-region age
    7. Track reset
    8. Memory retrieval

Not implemented:

    - feature extraction
    - appearance distance
    - association
    - Kalman
    - GMC


Core update
-----------

For visible region:

    M_t =
        alpha * M_(t-1)
        +
        (1-alpha) * A_t


For missing region:

    M_t =
        M_(t-1)


Aging
-----

When a region has not been observed for Δt frames:

    w_age =
        exp(-lambda * Δt)

The aged memory is:

    M_aged =
        normalize(
            M * w_age
        )

Important
---------

Aging is represented separately from the descriptor.

This avoids permanently destroying the stored
appearance just because a track was temporarily occluded.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# ============================================================================
# Utility
# ============================================================================


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


# ============================================================================
# Region memory
# ============================================================================


class RegionAppearanceMemory:
    """
    Temporal region appearance memory.

    Parameters
    ----------

    enable_appearance_memory:
        Main memory switch.

    enable_memory_aging:
        Enable age-based memory weight.

    memory_alpha:
        EMA coefficient.

        0.90 means:

            90% old memory
            10% new appearance

    memory_decay_lambda:
        Aging coefficient.

        weight =
            exp(-lambda * age)

    min_update_visibility:
        Region must have at least this visibility
        to update memory.

    region_names:
        Region list.

    normalize_memory:
        L2 normalize after EMA.

    verbose:
        Print configuration.
    """

    DEFAULT_REGIONS = (
        "head",
        "torso",
        "legs",
    )

    def __init__(
        self,

        enable_appearance_memory: bool = True,

        enable_memory_aging: bool = True,

        memory_alpha: float = 0.90,

        memory_decay_lambda: float = 0.05,

        min_update_visibility: float = 0.25,

        region_names: Sequence[str] = (
            "head",
            "torso",
            "legs",
        ),

        normalize_memory: bool = True,

        verbose: bool = False,
    ):

        self.enable_appearance_memory = bool(
            enable_appearance_memory
        )

        self.enable_memory_aging = bool(
            enable_memory_aging
        )

        self.memory_alpha = float(
            memory_alpha
        )

        self.memory_decay_lambda = float(
            memory_decay_lambda
        )

        self.min_update_visibility = float(
            min_update_visibility
        )

        self.region_names = tuple(
            region_names
        )

        self.normalize_memory = bool(
            normalize_memory
        )

        self.verbose = bool(
            verbose
        )

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

        if not (
            0.0
            <= self.memory_alpha
            <= 1.0
        ):

            raise ValueError(
                "memory_alpha must be in [0,1]"
            )

        if self.memory_decay_lambda < 0:

            raise ValueError(
                "memory_decay_lambda "
                "must be >= 0"
            )

        if not (
            0.0
            <= self.min_update_visibility
            <= 1.0
        ):

            raise ValueError(
                "min_update_visibility "
                "must be in [0,1]"
            )

        if len(
            self.region_names
        ) == 0:

            raise ValueError(
                "region_names cannot be empty"
            )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------

        self.memory = {}

        self.last_update_frame = {}

        self.age = {}

        self.update_count = {}

        self.initialized = {}

        self.reset()

        if self.verbose:

            self.print_config()

    # ======================================================================
    # Reset
    # ======================================================================

    def reset(self):

        self.memory = {
            region_name: None
            for region_name
            in self.region_names
        }

        self.last_update_frame = {
            region_name: None
            for region_name
            in self.region_names
        }

        self.age = {
            region_name: 0
            for region_name
            in self.region_names
        }

        self.update_count = {
            region_name: 0
            for region_name
            in self.region_names
        }

        self.initialized = {
            region_name: False
            for region_name
            in self.region_names
        }

    # ======================================================================
    # Configuration
    # ======================================================================

    def get_config(self) -> Dict:

        return {
            "enable_appearance_memory":
                self.enable_appearance_memory,

            "enable_memory_aging":
                self.enable_memory_aging,

            "memory_alpha":
                self.memory_alpha,

            "memory_decay_lambda":
                self.memory_decay_lambda,

            "min_update_visibility":
                self.min_update_visibility,

            "region_names":
                self.region_names,

            "normalize_memory":
                self.normalize_memory,
        }

    def print_config(self):

        print(
            "\n"
            "============================================================\n"
            "RegionAppearanceMemory\n"
            "============================================================\n"
            f"appearance_memory       : "
            f"{self.enable_appearance_memory}\n"

            f"memory_aging            : "
            f"{self.enable_memory_aging}\n"

            f"memory_alpha            : "
            f"{self.memory_alpha}\n"

            f"memory_decay_lambda     : "
            f"{self.memory_decay_lambda}\n"

            f"min_update_visibility   : "
            f"{self.min_update_visibility}\n"

            f"regions                 : "
            f"{self.region_names}\n"

            "============================================================\n"
        )

    # ======================================================================
    # Validation
    # ======================================================================

    def _check_region(
        self,
        region_name: str,
    ):

        if region_name not in self.region_names:

            raise KeyError(
                f"Unknown region "
                f"'{region_name}'. "
                f"Available: "
                f"{self.region_names}"
            )

    # ======================================================================
    # Visibility
    # ======================================================================

    def _get_visibility(
        self,
        region_name: str,
        visibility,
        metadata: Optional[Dict] = None,
    ) -> float:
        """
        Resolve region visibility.
        """

        if visibility is not None:

            try:

                value = float(
                    visibility
                )

            except (
                TypeError,
                ValueError,
            ):

                value = 0.0

            return float(
                np.clip(
                    value,
                    0.0,
                    1.0,
                )
            )

        if metadata is not None:

            region_meta = metadata.get(
                region_name,
                {},
            )

            if "visibility" in region_meta:

                try:

                    value = float(
                        region_meta[
                            "visibility"
                        ]
                    )

                    return float(
                        np.clip(
                            value,
                            0.0,
                            1.0,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

        return 1.0

    # ======================================================================
    # Update region
    # ======================================================================

    def update_region(
        self,
        region_name: str,
        feature: Optional[np.ndarray],
        visibility: Optional[float] = None,
        frame_index: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Update one region.

        Returns update information.
        """

        self._check_region(
            region_name
        )

        # ------------------------------------------------------------------
        # Memory disabled
        # ------------------------------------------------------------------

        if not self.enable_appearance_memory:

            return {
                "updated": False,
                "reason": "memory_disabled",
                "visibility": 0.0,
                "age": self.age[
                    region_name
                ],
            }

        # ------------------------------------------------------------------
        # Invalid feature
        # ------------------------------------------------------------------

        if feature is None:

            self._advance_age(
                region_name,
                frame_index,
            )

            return {
                "updated": False,
                "reason": "feature_missing",
                "visibility": 0.0,
                "age": self.age[
                    region_name
                ],
            }

        feature = np.asarray(
            feature,
            dtype=np.float32,
        ).reshape(-1)

        if feature.size == 0:

            self._advance_age(
                region_name,
                frame_index,
            )

            return {
                "updated": False,
                "reason": "empty_feature",
                "visibility": 0.0,
                "age": self.age[
                    region_name
                ],
            }

        # ------------------------------------------------------------------
        # Visibility
        # ------------------------------------------------------------------

        region_visibility = (
            self._get_visibility(
                region_name=region_name,
                visibility=visibility,
                metadata=metadata,
            )
        )

        # ------------------------------------------------------------------
        # Too little visible
        # ------------------------------------------------------------------

        if (
            region_visibility
            < self.min_update_visibility
        ):

            self._advance_age(
                region_name,
                frame_index,
            )

            return {
                "updated": False,
                "reason": "insufficient_visibility",
                "visibility": region_visibility,
                "age": self.age[
                    region_name
                ],
            }

        # ------------------------------------------------------------------
        # Normalize incoming feature
        # ------------------------------------------------------------------

        if self.normalize_memory:

            feature = _l2_normalize(
                feature
            )

        # ------------------------------------------------------------------
        # First observation
        # ------------------------------------------------------------------

        if not self.initialized[
            region_name
        ]:

            self.memory[
                region_name
            ] = feature.copy()

            self.initialized[
                region_name
            ] = True

            self.update_count[
                region_name
            ] = 1

            self.age[
                region_name
            ] = 0

            self.last_update_frame[
                region_name
            ] = frame_index

            return {
                "updated": True,
                "reason": "initialized",
                "visibility": region_visibility,
                "age": 0,
            }

        # ------------------------------------------------------------------
        # EMA
        # ------------------------------------------------------------------

        old_memory = self.memory[
            region_name
        ]

        if old_memory.shape != feature.shape:

            raise ValueError(
                f"Feature dimension changed "
                f"for region '{region_name}': "
                f"{old_memory.shape} -> "
                f"{feature.shape}"
            )

        alpha = self.memory_alpha

        new_memory = (
            alpha
            * old_memory
            +
            (1.0 - alpha)
            * feature
        )

        if self.normalize_memory:

            new_memory = _l2_normalize(
                new_memory
            )

        self.memory[
            region_name
        ] = new_memory

        self.update_count[
            region_name
        ] += 1

        self.age[
            region_name
        ] = 0

        self.last_update_frame[
            region_name
        ] = frame_index

        return {
            "updated": True,
            "reason": "ema",
            "visibility": region_visibility,
            "age": 0,
        }

    # ======================================================================
    # Update all regions
    # ======================================================================

    def update(
        self,
        region_features: Dict[str, np.ndarray],
        region_metadata: Optional[
            Dict[str, Dict]
        ] = None,
        frame_index: Optional[int] = None,
    ) -> Dict:
        """
        Update all regions.

        Missing regions are intentionally NOT replaced
        by zeros.

        Their previous memory is preserved.
        """

        results = {}

        for region_name in self.region_names:

            feature = region_features.get(
                region_name
            )

            metadata = None

            if region_metadata is not None:

                metadata = region_metadata.get(
                    region_name
                )

            visibility = None

            if metadata is not None:

                visibility = metadata.get(
                    "visibility"
                )

            results[
                region_name
            ] = self.update_region(
                region_name=region_name,
                feature=feature,
                visibility=visibility,
                frame_index=frame_index,
                metadata=region_metadata,
            )

        return results

    # ======================================================================
    # Age handling
    # ======================================================================

    def _advance_age(
        self,
        region_name: str,
        frame_index: Optional[int],
    ):

        if frame_index is None:

            self.age[
                region_name
            ] += 1

            return

        last_frame = (
            self.last_update_frame[
                region_name
            ]
        )

        if last_frame is None:

            self.age[
                region_name
            ] += 1

        else:

            self.age[
                region_name
            ] = max(
                0,
                int(
                    frame_index
                    -
                    last_frame
                ),
            )

    # ======================================================================
    # Explicit aging
    # ======================================================================

    def age_memory(
        self,
        frame_index: Optional[int] = None,
        delta_frames: Optional[int] = None,
    ):
        """
        Advance memory age.

        The descriptor itself is not destroyed.
        """

        for region_name in self.region_names:

            if delta_frames is not None:

                self.age[
                    region_name
                ] += max(
                    0,
                    int(delta_frames)
                )

            elif frame_index is not None:

                last_frame = (
                    self.last_update_frame[
                        region_name
                    ]
                )

                if last_frame is None:

                    self.age[
                        region_name
                    ] += 1

                else:

                    self.age[
                        region_name
                    ] = max(
                        0,
                        int(
                            frame_index
                            -
                            last_frame
                        ),
                    )

            else:

                self.age[
                    region_name
                ] += 1

    # ======================================================================
    # Aging weight
    # ======================================================================

    def get_age_weight(
        self,
        region_name: str,
    ) -> float:

        self._check_region(
            region_name
        )

        if not self.enable_memory_aging:

            return 1.0

        age = float(
            self.age[
                region_name
            ]
        )

        return float(
            np.exp(
                -self.memory_decay_lambda
                *
                age
            )
        )

    # ======================================================================
    # Get memory
    # ======================================================================

    def get(
        self,
        region_name: str,
        apply_aging: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get stored region appearance.

        If aging is enabled, return the same
        normalized direction but expose aging
        through separate weight.

        This avoids changing descriptor geometry.
        """

        self._check_region(
            region_name
        )

        feature = self.memory[
            region_name
        ]

        if feature is None:

            return None

        feature = feature.copy()

        # ------------------------------------------------------------------
        # We intentionally DO NOT multiply the descriptor
        # by age here.
        #
        # Aging is handled as a weight.
        # ------------------------------------------------------------------

        return feature

    # ======================================================================
    # Get memory + age
    # ======================================================================

    def get_with_age(
        self,
        region_name: str,
    ):
        """
        Return:

            feature
            age
            age_weight
        """

        feature = self.get(
            region_name
        )

        return (
            feature,
            self.age[
                region_name
            ],
            self.get_age_weight(
                region_name
            ),
        )

    # ======================================================================
    # Get all
    # ======================================================================

    def get_all(
        self,
        apply_aging: bool = True,
    ) -> Dict[str, Optional[np.ndarray]]:

        return {
            region_name:
                self.get(
                    region_name,
                    apply_aging=apply_aging,
                )

            for region_name
            in self.region_names
        }

    # ======================================================================
    # Metadata
    # ======================================================================

    def get_metadata(self) -> Dict:

        output = {}

        for region_name in self.region_names:

            output[
                region_name
            ] = {
                "initialized":
                    bool(
                        self.initialized[
                            region_name
                        ]
                    ),

                "age":
                    int(
                        self.age[
                            region_name
                        ]
                    ),

                "age_weight":
                    float(
                        self.get_age_weight(
                            region_name
                        )
                    ),

                "update_count":
                    int(
                        self.update_count[
                            region_name
                        ]
                    ),

                "last_update_frame":
                    self.last_update_frame[
                        region_name
                    ],
            }

        return output

    # ======================================================================
    # State
    # ======================================================================

    def state_dict(self) -> Dict:

        return {
            "memory": {
                name:
                    (
                        None
                        if self.memory[name]
                        is None
                        else self.memory[
                            name
                        ].copy()
                    )

                for name
                in self.region_names
            },

            "last_update_frame":
                dict(
                    self.last_update_frame
                ),

            "age":
                dict(
                    self.age
                ),

            "update_count":
                dict(
                    self.update_count
                ),

            "initialized":
                dict(
                    self.initialized
                ),
        }

    def load_state_dict(
        self,
        state: Dict,
    ):

        for region_name in self.region_names:

            if region_name in state.get(
                "memory",
                {},
            ):

                feature = state[
                    "memory"
                ][
                    region_name
                ]

                if feature is None:

                    self.memory[
                        region_name
                    ] = None

                else:

                    self.memory[
                        region_name
                    ] = np.asarray(
                        feature,
                        dtype=np.float32,
                    ).copy()

            if region_name in state.get(
                "last_update_frame",
                {},
            ):

                self.last_update_frame[
                    region_name
                ] = state[
                    "last_update_frame"
                ][
                    region_name
                ]

            if region_name in state.get(
                "age",
                {},
            ):

                self.age[
                    region_name
                ] = int(
                    state[
                        "age"
                    ][
                        region_name
                    ]
                )

            if region_name in state.get(
                "update_count",
                {},
            ):

                self.update_count[
                    region_name
                ] = int(
                    state[
                        "update_count"
                    ][
                        region_name
                    ]
                )

            if region_name in state.get(
                "initialized",
                {},
            ):

                self.initialized[
                    region_name
                ] = bool(
                    state[
                        "initialized"
                    ][
                        region_name
                    ]
                )


# ============================================================================
# Factory
# ============================================================================


def create_appearance_memory(
    enable_appearance_memory: bool = True,
    enable_memory_aging: bool = True,
    memory_alpha: float = 0.90,
    memory_decay_lambda: float = 0.05,
    min_update_visibility: float = 0.25,
):
    """
    Recommended factory.
    """

    return RegionAppearanceMemory(
        enable_appearance_memory=(
            enable_appearance_memory
        ),

        enable_memory_aging=(
            enable_memory_aging
        ),

        memory_alpha=(
            memory_alpha
        ),

        memory_decay_lambda=(
            memory_decay_lambda
        ),

        min_update_visibility=(
            min_update_visibility
        ),
    )


# ============================================================================
# Self-test
# ============================================================================


if __name__ == "__main__":

    print(
        "\n"
        "============================================================\n"
        "REGION APPEARANCE MEMORY SELF TEST\n"
        "============================================================"
    )

    rng = np.random.default_rng(
        42
    )

    memory = RegionAppearanceMemory(
        enable_appearance_memory=True,
        enable_memory_aging=True,
        memory_alpha=0.90,
        memory_decay_lambda=0.05,
        min_update_visibility=0.25,
    )

    # ------------------------------------------------------------------
    # Initial appearance
    # ------------------------------------------------------------------

    feature_head = rng.random(
        30
    ).astype(
        np.float32
    )

    feature_torso = rng.random(
        24
    ).astype(
        np.float32
    )

    feature_legs = rng.random(
        24
    ).astype(
        np.float32
    )

    region_features = {
        "head":
            feature_head,

        "torso":
            feature_torso,

        "legs":
            feature_legs,
    }

    region_metadata = {
        "head": {
            "visibility": 1.0,
        },

        "torso": {
            "visibility": 1.0,
        },

        "legs": {
            "visibility": 1.0,
        },
    }

    result = memory.update(
        region_features,
        region_metadata,
        frame_index=1,
    )

    print(
        "\nInitial update:"
    )

    print(
        result
    )

    assert memory.initialized[
        "head"
    ]

    # ------------------------------------------------------------------
    # Occlusion
    # ------------------------------------------------------------------

    result = memory.update(
        {
            "head":
                feature_head,

            "torso":
                None,

            "legs":
                None,
        },

        {
            "head": {
                "visibility": 1.0,
            },

            "torso": {
                "visibility": 0.0,
            },

            "legs": {
                "visibility": 0.0,
            },
        },

        frame_index=2,
    )

    print(
        "\nOcclusion update:"
    )

    print(
        result
    )

    # Memory should still exist.
    assert memory.get(
        "torso"
    ) is not None

    assert memory.get(
        "legs"
    ) is not None

    # ------------------------------------------------------------------
    # Aging
    # ------------------------------------------------------------------

    memory.age_memory(
        delta_frames=10
    )

    print(
        "\nAfter 10 frames:"
    )

    print(
        memory.get_metadata()
    )

    head_weight = (
        memory.get_age_weight(
            "head"
        )
    )

    torso_weight = (
        memory.get_age_weight(
            "torso"
        )
    )

    print(
        "Head age weight:",
        head_weight,
    )

    print(
        "Torso age weight:",
        torso_weight,
    )

    assert (
        0.0
        < torso_weight
        <= 1.0
    )

    print(
        "\n"
        "============================================================\n"
        "SELF TEST PASSED\n"
        "============================================================"
    )