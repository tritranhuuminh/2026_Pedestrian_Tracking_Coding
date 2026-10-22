from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


# ============================================================================
# Utility
# ============================================================================

def _l2_normalize(
    feature: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:

    feature = np.asarray(feature, dtype=np.float32)

    norm = float(np.linalg.norm(feature))

    if norm <= eps:
        return np.zeros_like(feature, dtype=np.float32)

    return (feature / norm).astype(np.float32)


# ============================================================================
# KeypointPatchAppearanceExtractor
# ============================================================================

class KeypointPatchAppearanceExtractor:

    DEFAULT_REGION_INDICES = {
        "head": (0, 1, 2, 3, 4),
        "torso": (5, 6, 11, 12),
        "legs": (13, 14, 15, 16),
    }

    REGION_ORDER = (
        "head",
        "torso",
        "legs",
    )

    RGB_DIM = 6
    HSV_DIM = 6

    def __init__(
        self,
        patch_radius: int = 8,
        resize_size: int = 16,

        enable_bbox_normalized_patch: bool = False,
        patch_radius_ratio: float = 0.02,

        enable_hsv_appearance: bool = False,

        enable_multiscale_patch: bool = False,
        patch_scales: Sequence[float] = (0.5, 1.0, 2.0),
        scale_weights: Optional[Sequence[float]] = None,

        enable_region_appearance: bool = False,

        enable_confidence_weighting: bool = False,
        confidence_power: float = 1.0,

        use_confidence_for_validity: bool = False,
        min_keypoint_confidence: float = 0.30,

        enable_adaptive_confidence_threshold: bool = False,
        adaptive_otsu_bins: int = 64,
        adaptive_floor_ratio: float = 0.3,
        adaptive_min_samples: int = 30,
        adaptive_ema_alpha: float = 0.9,

        keypoint_indices: Optional[Sequence[int]] = None,

        auto_normalize_coordinates: bool = True,

        region_indices: Optional[
            Dict[str, Sequence[int]]
        ] = None,
    ):

        self.patch_radius = int(patch_radius)
        self.resize_size = int(resize_size)

        self.enable_bbox_normalized_patch = bool(
            enable_bbox_normalized_patch
        )

        self.patch_radius_ratio = float(
            patch_radius_ratio
        )

        self.enable_hsv_appearance = bool(
            enable_hsv_appearance
        )

        self.enable_multiscale_patch = bool(
            enable_multiscale_patch
        )

        self.patch_scales = tuple(
            float(x) for x in patch_scales
        )

        if not self.patch_scales:
            self.patch_scales = (1.0,)

        if scale_weights is None:
            self.scale_weights = tuple(
                1.0 for _ in self.patch_scales
            )
        else:
            self.scale_weights = tuple(
                float(x) for x in scale_weights
            )

        self.enable_region_appearance = bool(
            enable_region_appearance
        )

        self.enable_confidence_weighting = bool(
            enable_confidence_weighting
        )

        self.confidence_power = float(
            confidence_power
        )

        self.use_confidence_for_validity = bool(
            use_confidence_for_validity
        )

        self.min_keypoint_confidence = float(
            min_keypoint_confidence
        )

        # --------------------------------------------------------------
        # Adaptive (Otsu) confidence threshold
        # --------------------------------------------------------------

        self.enable_adaptive_confidence_threshold = bool(
            enable_adaptive_confidence_threshold
        )

        self.adaptive_otsu_bins = int(
            adaptive_otsu_bins
        )

        self.adaptive_floor_ratio = float(
            adaptive_floor_ratio
        )

        self.adaptive_min_samples = int(
            adaptive_min_samples
        )

        self.adaptive_ema_alpha = float(
            adaptive_ema_alpha
        )

        # Running (EMA-smoothed) adaptive threshold, carried across
        # successive calls to `extract()` so the threshold does not
        # jump frame-to-frame (which would otherwise flip keypoints
        # between valid/invalid and destabilize the appearance
        # descriptor used for tracking / ReID).
        self._adaptive_threshold_ema: Optional[float] = None

        if keypoint_indices is None:
            self.keypoint_indices = None
        else:
            self.keypoint_indices = tuple(
                int(i) for i in keypoint_indices
            )

        self.auto_normalize_coordinates = bool(
            auto_normalize_coordinates
        )

        if region_indices is None:
            self.region_indices = {
                name: tuple(indices)
                for name, indices
                in self.DEFAULT_REGION_INDICES.items()
            }
        else:
            self.region_indices = {}

            for name in self.REGION_ORDER:

                if name not in region_indices:
                    raise ValueError(
                        f"Missing region '{name}'."
                    )

                self.region_indices[name] = tuple(
                    int(i)
                    for i in region_indices[name]
                )

        self._validate()

    # ========================================================================
    # Validation
    # ========================================================================

    def _validate(self):

        if self.patch_radius < 0:
            raise ValueError(
                "patch_radius must be >= 0"
            )

        if self.resize_size <= 0:
            raise ValueError(
                "resize_size must be > 0"
            )

        if self.patch_radius_ratio <= 0:
            raise ValueError(
                "patch_radius_ratio must be > 0"
            )

        if any(
            scale <= 0
            for scale in self.patch_scales
        ):
            raise ValueError(
                "patch_scales must contain positive values"
            )

        if len(self.scale_weights) != len(
            self.patch_scales
        ):
            raise ValueError(
                "scale_weights and patch_scales "
                "must have the same length"
            )

        if any(
            weight < 0
            for weight in self.scale_weights
        ):
            raise ValueError(
                "scale_weights must be >= 0"
            )

        if self.confidence_power <= 0:
            raise ValueError(
                "confidence_power must be > 0"
            )

        if not (
            0.0
            <= self.min_keypoint_confidence
            <= 1.0
        ):
            raise ValueError(
                "min_keypoint_confidence must be in [0, 1]"
            )

        if self.adaptive_otsu_bins < 2:
            raise ValueError(
                "adaptive_otsu_bins must be >= 2"
            )

        if not (
            0.0
            <= self.adaptive_floor_ratio
            <= 1.0
        ):
            raise ValueError(
                "adaptive_floor_ratio must be in [0, 1]"
            )

        if self.adaptive_min_samples < 1:
            raise ValueError(
                "adaptive_min_samples must be >= 1"
            )

        if not (
            0.0
            <= self.adaptive_ema_alpha
            < 1.0
        ):
            raise ValueError(
                "adaptive_ema_alpha must be in [0, 1)"
            )

    # ========================================================================
    # Descriptor dimensions
    # ========================================================================

    @property
    def base_descriptor_dim(self) -> int:

        if self.enable_hsv_appearance:
            return self.RGB_DIM + self.HSV_DIM

        return self.RGB_DIM

    @property
    def number_of_scales(self) -> int:

        if self.enable_multiscale_patch:
            return len(self.patch_scales)

        return 1

    @property
    def descriptor_dim_per_keypoint(self) -> int:

        return (
            self.base_descriptor_dim
            * self.number_of_scales
        )

    # ========================================================================
    # Keypoint preparation
    # ========================================================================

    def _prepare_keypoints(
        self,
        keypoints,
        image_shape: Tuple,
    ):

        if hasattr(keypoints, "detach"):

            keypoints = (
                keypoints
                .detach()
                .cpu()
                .numpy()
            )

        keypoints = np.asarray(
            keypoints,
            dtype=np.float32,
        )

        if keypoints.ndim == 2:

            if keypoints.shape[1] not in (2, 3):
                raise ValueError(
                    "Expected Kx2 or Kx3 keypoints"
                )

            keypoints = keypoints[None, ...]

        elif keypoints.ndim == 3:

            if keypoints.shape[2] not in (2, 3):
                raise ValueError(
                    "Expected NxKx2 or NxKx3 keypoints"
                )

        else:

            raise ValueError(
                "Invalid keypoint shape"
            )

        coords = keypoints[:, :, :2].copy()

        confidence = None

        if keypoints.shape[2] == 3:

            confidence = keypoints[:, :, 2].copy()

            confidence = np.nan_to_num(
                confidence,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            confidence = np.clip(
                confidence,
                0.0,
                1.0,
            )

        if self.auto_normalize_coordinates:

            coords = self._maybe_denormalize_coordinates(
                coords,
                image_shape,
            )

        return coords, confidence

    # ========================================================================
    # Coordinates
    # ========================================================================

    def _maybe_denormalize_coordinates(
        self,
        coords: np.ndarray,
        image_shape: Tuple,
    ):

        finite = np.isfinite(coords)

        if not np.any(finite):
            return coords

        values = coords[finite]

        if values.size == 0:
            return coords

        if (
            np.min(values) >= -0.05
            and np.max(values) <= 1.05
        ):

            height = image_shape[0]
            width = image_shape[1]

            coords = coords.copy()

            coords[:, :, 0] *= width - 1
            coords[:, :, 1] *= height - 1

        return coords

    # ========================================================================
    # Patch radius
    # ========================================================================

    def _calculate_patch_radius(
        self,
        bbox: Optional[Sequence[float]],
        scale: float = 1.0,
    ) -> int:

        if (
            self.enable_bbox_normalized_patch
            and bbox is not None
        ):

            try:

                x1, y1, x2, y2 = [
                    float(v)
                    for v in bbox[:4]
                ]

                bbox_height = abs(y2 - y1)

                radius = (
                    bbox_height
                    * self.patch_radius_ratio
                )

            except (
                TypeError,
                ValueError,
                IndexError,
            ):

                radius = float(
                    self.patch_radius
                )

        else:

            radius = float(
                self.patch_radius
            )

        radius *= float(scale)

        return max(
            0,
            int(round(radius)),
        )

    # ========================================================================
    # Adaptive (Otsu) confidence threshold
    # ========================================================================

    def _otsu_threshold(
        self,
        values: np.ndarray,
    ) -> float:
        """
        Find the confidence value that best splits `values` into a
        "low-confidence" cluster and a "high-confidence" cluster by
        maximizing between-class variance (Otsu's method).
        """

        bins = self.adaptive_otsu_bins

        hist, edges = np.histogram(
            values,
            bins=bins,
            range=(0.0, 1.0),
        )

        hist = hist.astype(np.float64)
        total = float(hist.sum())

        if total <= 0:
            return self.min_keypoint_confidence

        centers = (edges[:-1] + edges[1:]) / 2.0
        sum_all = float(np.dot(hist, centers))

        weight_bg = 0.0
        sum_bg = 0.0

        best_thr = self.min_keypoint_confidence
        best_between_var = -1.0

        for i in range(bins):

            weight_bg += hist[i]
            sum_bg += hist[i] * centers[i]

            if weight_bg <= 0:
                continue

            weight_fg = total - weight_bg

            if weight_fg <= 0:
                break

            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_all - sum_bg) / weight_fg

            between_var = (
                weight_bg
                * weight_fg
                * (mean_bg - mean_fg) ** 2
            )

            if between_var > best_between_var:
                best_between_var = between_var
                best_thr = float(centers[i])

        return best_thr

    def reset_adaptive_state(self) -> None:
        """
        Clear the EMA-smoothed adaptive threshold memory. Call this when
        starting a new video / sequence / track session so state from a
        previous, unrelated sequence does not leak in as the initial
        threshold.
        """

        self._adaptive_threshold_ema = None

    def _compute_adaptive_confidence_threshold(
        self,
        confidence: Optional[np.ndarray],
    ) -> float:
        """
        Compute a confidence threshold from ALL confidence values in the
        current call (every person, every keypoint) using Otsu's method,
        then stabilize it for tracking use:

          1. Sample-size guard: if too few confidence values are present
             in this call (e.g. a single person's 17 keypoints), Otsu is
             statistically unreliable and prone to picking an arbitrary
             split point that drifts frame-to-frame. In that case fall
             back to the fixed `min_keypoint_confidence` for this call.
          2. Floor: never let the threshold drop so low that it treats
             pure noise as valid (e.g. when the whole scene is occluded).
          3. Temporal EMA smoothing: blend the new raw estimate with the
             running threshold so it changes gradually across frames
             instead of jumping. Sudden jumps flip keypoints between
             valid/invalid, which changes the L2-normalized descriptor's
             direction abruptly and hurts appearance-based re-ID matching
             (manifesting as more ID switches in trackers like BoT-SORT).
        """

        if confidence is None:
            return self.min_keypoint_confidence

        values = np.asarray(
            confidence,
            dtype=np.float32,
        ).reshape(-1)

        values = values[np.isfinite(values)]

        if values.size < self.adaptive_min_samples:

            raw_threshold = self.min_keypoint_confidence

        else:

            raw_threshold = self._otsu_threshold(
                values
            )

        floor = (
            self.min_keypoint_confidence
            * self.adaptive_floor_ratio
        )

        raw_threshold = max(raw_threshold, floor)

        raw_threshold = float(
            np.clip(raw_threshold, 0.0, 1.0)
        )

        if self._adaptive_threshold_ema is None:

            self._adaptive_threshold_ema = raw_threshold

        else:

            alpha = self.adaptive_ema_alpha

            self._adaptive_threshold_ema = (
                alpha * self._adaptive_threshold_ema
                + (1.0 - alpha) * raw_threshold
            )

        return float(
            np.clip(
                self._adaptive_threshold_ema,
                0.0,
                1.0,
            )
        )

    # ========================================================================
    # Keypoint validity
    # ========================================================================

    def _is_valid_keypoint(
        self,
        x: float,
        y: float,
        confidence: Optional[float],
        image_width: int,
        image_height: int,
        patch_radius: int,
        confidence_threshold: float,
    ) -> bool:

        if not np.isfinite(x):
            return False

        if not np.isfinite(y):
            return False

        margin = float(patch_radius)

        if x < -margin:
            return False

        if x >= image_width + margin:
            return False

        if y < -margin:
            return False

        if y >= image_height + margin:
            return False

        if (
            self.use_confidence_for_validity
            and confidence is not None
        ):

            if confidence < confidence_threshold:
                return False

        return True

    # ========================================================================
    # Patch extraction
    # ========================================================================

    def _extract_patch(
        self,
        image: np.ndarray,
        x: float,
        y: float,
        radius: int,
    ) -> Optional[np.ndarray]:

        if image is None:
            return None

        if image.ndim != 3:
            return None

        height, width = image.shape[:2]

        cx = int(round(x))
        cy = int(round(y))

        radius = max(
            0,
            int(radius),
        )

        x1 = max(
            0,
            cx - radius,
        )

        y1 = max(
            0,
            cy - radius,
        )

        x2 = min(
            width,
            cx + radius + 1,
        )

        y2 = min(
            height,
            cy + radius + 1,
        )

        if x2 <= x1 or y2 <= y1:
            return None

        patch = image[
            y1:y2,
            x1:x2,
        ]

        if patch.size == 0:
            return None

        patch = cv2.cvtColor(
            patch,
            cv2.COLOR_BGR2RGB,
        )

        patch = cv2.resize(
            patch,
            (
                self.resize_size,
                self.resize_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return patch

    # ========================================================================
    # RGB descriptor
    # ========================================================================

    def _rgb_descriptor(
        self,
        patch_rgb: np.ndarray,
    ) -> np.ndarray:

        patch = (
            patch_rgb.astype(np.float32)
            / 255.0
        )

        mean = np.mean(
            patch,
            axis=(0, 1),
        )

        std = np.std(
            patch,
            axis=(0, 1),
        )

        return np.concatenate(
            [mean, std],
            axis=0,
        ).astype(np.float32)

    # ========================================================================
    # HSV descriptor
    # ========================================================================

    def _hsv_descriptor(
        self,
        patch_rgb: np.ndarray,
    ) -> np.ndarray:

        hsv = cv2.cvtColor(
            patch_rgb,
            cv2.COLOR_RGB2HSV,
        ).astype(np.float32)

        hsv[:, :, 0] /= 179.0
        hsv[:, :, 1] /= 255.0
        hsv[:, :, 2] /= 255.0

        mean = np.mean(
            hsv,
            axis=(0, 1),
        )

        std = np.std(
            hsv,
            axis=(0, 1),
        )

        return np.concatenate(
            [mean, std],
            axis=0,
        ).astype(np.float32)

    # ========================================================================
    # Single-scale descriptor
    # ========================================================================

    def _single_scale_descriptor(
        self,
        patch_rgb: np.ndarray,
    ) -> np.ndarray:

        descriptor = [
            self._rgb_descriptor(
                patch_rgb
            )
        ]

        if self.enable_hsv_appearance:

            descriptor.append(
                self._hsv_descriptor(
                    patch_rgb
                )
            )

        descriptor = np.concatenate(
            descriptor,
            axis=0,
        )

        return _l2_normalize(
            descriptor
        )

    # ========================================================================
    # Confidence
    # ========================================================================

    def _get_confidence_weight(
        self,
        confidence: Optional[float],
    ) -> float:

        if not self.enable_confidence_weighting:
            return 1.0

        if confidence is None:
            return 1.0

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

        return float(
            confidence ** self.confidence_power
        )

    # ========================================================================
    # Keypoint descriptor
    # ========================================================================

    def _extract_single_keypoint_descriptor(
        self,
        image: np.ndarray,
        x: float,
        y: float,
        confidence: Optional[float],
        bbox: Optional[Sequence[float]],
        confidence_threshold: float,
    ):

        image_height, image_width = image.shape[:2]

        if self.enable_multiscale_patch:

            scales = self.patch_scales
            weights = self.scale_weights

        else:

            scales = (1.0,)
            weights = (1.0,)

        descriptors = []
        any_valid = False

        for scale_idx, scale in enumerate(scales):

            radius = self._calculate_patch_radius(
                bbox=bbox,
                scale=scale,
            )

            valid = self._is_valid_keypoint(
                x=x,
                y=y,
                confidence=confidence,
                image_width=image_width,
                image_height=image_height,
                patch_radius=radius,
                confidence_threshold=confidence_threshold,
            )

            if not valid:

                descriptor = np.zeros(
                    self.base_descriptor_dim,
                    dtype=np.float32,
                )

            else:

                patch = self._extract_patch(
                    image=image,
                    x=x,
                    y=y,
                    radius=radius,
                )

                if patch is None:

                    descriptor = np.zeros(
                        self.base_descriptor_dim,
                        dtype=np.float32,
                    )

                else:

                    descriptor = (
                        self._single_scale_descriptor(
                            patch
                        )
                    )

                    any_valid = True

            descriptor *= float(
                weights[scale_idx]
            )

            descriptors.append(
                descriptor
            )

        if not any_valid:

            return (
                np.zeros(
                    self.descriptor_dim_per_keypoint,
                    dtype=np.float32,
                ),
                False,
            )

        descriptor = np.concatenate(
            descriptors,
            axis=0,
        )

        descriptor = _l2_normalize(
            descriptor
        )

        descriptor *= self._get_confidence_weight(
            confidence
        )

        return (
            descriptor.astype(np.float32),
            True,
        )

    # ========================================================================
    # Region descriptor
    # ========================================================================

    def _extract_single_region(
        self,
        image: np.ndarray,
        coords: np.ndarray,
        confidence: Optional[np.ndarray],
        region_indices: Sequence[int],
        bbox: Optional[Sequence[float]],
        confidence_threshold: float,
    ):

        descriptors = []

        valid_mask = np.zeros(
            len(region_indices),
            dtype=bool,
        )

        valid_confidences = []

        for slot, kp_idx in enumerate(
            region_indices
        ):

            if (
                kp_idx < 0
                or kp_idx >= coords.shape[0]
            ):

                descriptors.append(
                    np.zeros(
                        self.descriptor_dim_per_keypoint,
                        dtype=np.float32,
                    )
                )

                continue

            x = float(
                coords[kp_idx, 0]
            )

            y = float(
                coords[kp_idx, 1]
            )

            kp_conf = None

            if confidence is not None:

                kp_conf = float(
                    confidence[kp_idx]
                )

            descriptor, valid = (
                self._extract_single_keypoint_descriptor(
                    image=image,
                    x=x,
                    y=y,
                    confidence=kp_conf,
                    bbox=bbox,
                    confidence_threshold=confidence_threshold,
                )
            )

            descriptors.append(
                descriptor
            )

            if valid:

                valid_mask[slot] = True

                valid_confidences.append(
                    1.0
                    if kp_conf is None
                    else float(
                        np.clip(
                            kp_conf,
                            0.0,
                            1.0,
                        )
                    )
                )

        feature = np.concatenate(
            descriptors,
            axis=0,
        )

        feature = _l2_normalize(
            feature
        )

        if valid_confidences:

            region_confidence = float(
                np.mean(
                    valid_confidences
                )
            )

        else:

            region_confidence = 0.0

        valid_count = int(
            np.sum(valid_mask)
        )

        return (
            feature,
            region_confidence,
            valid_count,
            valid_mask,
        )

    # ========================================================================
    # Main public API
    # ========================================================================

    def extract(
        self,
        image: np.ndarray,
        keypoints,
        bbox: Optional[Sequence[float]] = None,
        bboxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        bbox : one bbox, applied to every person in this call. Use this
            for the single-person case (batch_size == 1) — unchanged,
            backward-compatible behavior.
        bboxes : a sequence of per-person bboxes, one per person in
            `keypoints` (len(bboxes) must equal the batch size). Use
            this when extracting for several people in one batched
            call (recommended for tracking pipelines — see note below).

        Note for trackers (e.g. BoT-SORT): call this ONCE per frame with
        every detected person's keypoints stacked into one NxKx3 array
        (and `bboxes` as their per-person boxes), rather than once per
        detection. Calling per-detection starves the adaptive Otsu
        threshold of samples (only K keypoints instead of N*K) and,
        because the EMA state lives on this extractor instance, causes
        it to update once per detection instead of once per frame —
        both of which destabilize the appearance descriptor and can
        increase tracker ID switches.
        """

        coords, confidence = (
            self._prepare_keypoints(
                keypoints,
                image.shape,
            )
        )

        batch_size = coords.shape[0]
        num_keypoints = coords.shape[1]

        if bboxes is not None:

            if len(bboxes) != batch_size:
                raise ValueError(
                    "bboxes must have exactly one entry per "
                    f"person (got {len(bboxes)}, expected {batch_size})"
                )

            bbox_list = list(bboxes)

        else:

            bbox_list = [bbox] * batch_size

        # ------------------------------------------------------------
        # Resolve the confidence threshold to use for this call.
        # If adaptive mode is enabled, compute one Otsu threshold from
        # every confidence value available in this call (all people,
        # all keypoints) instead of using the fixed
        # `min_keypoint_confidence`.
        # ------------------------------------------------------------

        if (
            self.enable_adaptive_confidence_threshold
            and self.use_confidence_for_validity
        ):

            confidence_threshold = (
                self._compute_adaptive_confidence_threshold(
                    confidence
                )
            )

        else:

            confidence_threshold = (
                self.min_keypoint_confidence
            )

        if self.keypoint_indices is None:

            selected_indices = tuple(
                range(num_keypoints)
            )

        else:

            selected_indices = tuple(
                i
                for i in self.keypoint_indices
                if 0 <= i < num_keypoints
            )

        outputs = []

        for person_idx in range(batch_size):

            person_coords = coords[
                person_idx
            ]

            person_bbox = bbox_list[
                person_idx
            ]

            person_confidence = None

            if confidence is not None:

                person_confidence = confidence[
                    person_idx
                ]

            # ------------------------------------------------------------
            # Region appearance
            # ------------------------------------------------------------

            if self.enable_region_appearance:

                region_descriptors = []

                for region_name in self.REGION_ORDER:

                    indices = self.region_indices[
                        region_name
                    ]

                    descriptor, _, _, _ = (
                        self._extract_single_region(
                            image=image,
                            coords=person_coords,
                            confidence=person_confidence,
                            region_indices=indices,
                            bbox=person_bbox,
                            confidence_threshold=confidence_threshold,
                        )
                    )

                    region_descriptors.append(
                        descriptor
                    )

                feature = np.concatenate(
                    region_descriptors,
                    axis=0,
                )

            # ------------------------------------------------------------
            # Keypoint appearance
            # ------------------------------------------------------------

            else:

                descriptors = []

                for kp_idx in selected_indices:

                    x = float(
                        person_coords[
                            kp_idx,
                            0,
                        ]
                    )

                    y = float(
                        person_coords[
                            kp_idx,
                            1,
                        ]
                    )

                    kp_conf = None

                    if person_confidence is not None:

                        kp_conf = float(
                            person_confidence[
                                kp_idx
                            ]
                        )

                    descriptor, _ = (
                        self._extract_single_keypoint_descriptor(
                            image=image,
                            x=x,
                            y=y,
                            confidence=kp_conf,
                            bbox=person_bbox,
                            confidence_threshold=confidence_threshold,
                        )
                    )

                    descriptors.append(
                        descriptor
                    )

                if descriptors:

                    feature = np.concatenate(
                        descriptors,
                        axis=0,
                    )

                else:

                    feature = np.zeros(
                        0,
                        dtype=np.float32,
                    )

            # ------------------------------------------------------------
            # Final global normalization
            # ------------------------------------------------------------

            feature = _l2_normalize(
                feature
            )

            outputs.append(
                feature.astype(np.float32)
            )

        if batch_size == 1:

            return outputs[0]

        return np.stack(
            outputs,
            axis=0,
        ).astype(np.float32)


# ============================================================================
# Factory
# ============================================================================

def create_extractor(
    *,
    enable_bbox_normalized_patch=False,
    enable_hsv_appearance=False,
    enable_multiscale_patch=False,
    enable_region_appearance=False,
    enable_confidence_weighting=False,
    use_confidence_for_validity=False,

    patch_radius=8,
    patch_radius_ratio=0.02,
    resize_size=16,

    patch_scales=(0.5, 1.0, 2.0),
    scale_weights=None,

    confidence_power=1.0,
    min_keypoint_confidence=0.30,

    enable_adaptive_confidence_threshold=False,
    adaptive_otsu_bins=64,
    adaptive_floor_ratio=0.3,
    adaptive_min_samples=30,
    adaptive_ema_alpha=0.9,

    keypoint_indices=None,
):

    return KeypointPatchAppearanceExtractor(

        patch_radius=patch_radius,
        resize_size=resize_size,

        enable_bbox_normalized_patch=(
            enable_bbox_normalized_patch
        ),

        patch_radius_ratio=(
            patch_radius_ratio
        ),

        enable_hsv_appearance=(
            enable_hsv_appearance
        ),

        enable_multiscale_patch=(
            enable_multiscale_patch
        ),

        patch_scales=patch_scales,

        scale_weights=scale_weights,

        enable_region_appearance=(
            enable_region_appearance
        ),

        enable_confidence_weighting=(
            enable_confidence_weighting
        ),

        confidence_power=(
            confidence_power
        ),

        use_confidence_for_validity=(
            use_confidence_for_validity
        ),

        min_keypoint_confidence=(
            min_keypoint_confidence
        ),

        enable_adaptive_confidence_threshold=(
            enable_adaptive_confidence_threshold
        ),

        adaptive_otsu_bins=(
            adaptive_otsu_bins
        ),

        adaptive_floor_ratio=(
            adaptive_floor_ratio
        ),

        adaptive_min_samples=(
            adaptive_min_samples
        ),

        adaptive_ema_alpha=(
            adaptive_ema_alpha
        ),

        keypoint_indices=(
            keypoint_indices
        ),
    )