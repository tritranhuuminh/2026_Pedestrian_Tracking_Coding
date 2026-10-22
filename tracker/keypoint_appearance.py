"""
tracker/keypoint_appearance.py
==============================

Keypoint-Guided Appearance Representation

This module contains ONLY the appearance representation algorithm.

It does NOT contain:
    - Experiment definitions
    - E0-E7 configuration
    - Region memory management
    - Missing-region association policy
    - Visibility weighting policy
    - Memory aging policy
    - BoT-SORT association logic

The module provides:
    1. Keypoint patch appearance
    2. Skeleton appearance
    3. Hybrid appearance
    4. Anatomical region appearance
    5. Region feature extraction
    6. Cosine distance

The experiment configuration is defined separately in:
    tracker/experiment_config.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class KeypointAppearanceExtractor:
    """
    Keypoint-guided appearance extractor.

    Supported appearance modes
    ---------------------------
    patch
        Appearance descriptor extracted independently around each
        selected keypoint.

    skeleton
        Appearance descriptor extracted from the skeleton region.

    hybrid
        Concatenation of patch and skeleton descriptors.

    region
        Appearance descriptor extracted independently from:
            - head
            - torso
            - legs

    region_hybrid
        Concatenation of region and skeleton descriptors.

    Notes
    -----
    This class is intentionally independent from BoT-SORT association.
    It only extracts appearance features.
    """

    # ------------------------------------------------------------
    # COCO keypoint definitions
    # ------------------------------------------------------------

    KEYPOINT_NAMES = [
        "nose",          # 0
        "left_eye",      # 1
        "right_eye",     # 2
        "left_ear",      # 3
        "right_ear",     # 4
        "left_shoulder", # 5
        "right_shoulder",# 6
        "left_elbow",    # 7
        "right_elbow",   # 8
        "left_wrist",    # 9
        "right_wrist",   # 10
        "left_hip",      # 11
        "right_hip",     # 12
        "left_knee",     # 13
        "right_knee",    # 14
        "left_ankle",    # 15
        "right_ankle",   # 16
    ]

    # Keypoints used for local appearance extraction.
    SELECTED_KEYPOINTS = [
        0,   # nose
        5,   # left shoulder
        6,   # right shoulder
        7,   # left elbow
        8,   # right elbow
        11,  # left hip
        12,  # right hip
        13,  # left knee
        14,  # right knee
        15,  # left ankle
        16,  # right ankle
    ]

    # ------------------------------------------------------------
    # Skeleton topology
    # ------------------------------------------------------------

    SKELETON_CONNECTIONS = [
        (0, 5),
        (0, 6),
        (5, 6),
        (5, 7),
        (6, 8),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]

    # ------------------------------------------------------------
    # Anatomical regions
    # ------------------------------------------------------------

    REGION_KEYPOINTS = {
        "head": [
            0, 1, 2, 3, 4
        ],

        "torso": [
            5, 6, 11, 12
        ],

        "legs": [
            11, 12, 13, 14, 15, 16
        ],
    }

    REGION_CONNECTIONS = {
        "head": [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
        ],

        "torso": [
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),
        ],

        "legs": [
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 12),
        ],
    }

    REGION_NAMES = (
        "head",
        "torso",
        "legs",
    )

    # ------------------------------------------------------------
    # Descriptor dimensions
    # ------------------------------------------------------------

    HSV_H_BINS = 16
    HSV_S_BINS = 8
    HSV_V_BINS = 8

    RGB_STAT_DIM = 6

    LOCAL_FEATURE_DIM = (
        HSV_H_BINS
        + HSV_S_BINS
        + HSV_V_BINS
        + RGB_STAT_DIM
    )

    # ------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------

    def __init__(
        self,
        appearance_mode: str = "patch",
        patch_ratio: float = 0.04,
        min_keypoint_conf: float = 0.25,
        use_keypoint_confidence: bool = True,
        use_region_confidence: bool = False,
        region_dilation_ratio: float = 0.03,
    ):
        """
        Parameters
        ----------
        appearance_mode:
            One of:
                patch
                skeleton
                hybrid
                region
                region_hybrid

        patch_ratio:
            Local patch size relative to person bounding-box size.

        min_keypoint_conf:
            Minimum confidence required for a keypoint to be used.

        use_keypoint_confidence:
            If True, keypoint confidence is used when aggregating
            patch appearance features.

        use_region_confidence:
            If True, region feature extraction applies the region
            confidence to the local feature.

            This is deliberately separate from BoT-SORT's
            region-level visibility weighting.

        region_dilation_ratio:
            Morphological dilation ratio for region masks.
        """

        valid_modes = {
            "patch",
            "skeleton",
            "hybrid",
            "region",
            "region_hybrid",
        }

        if appearance_mode not in valid_modes:
            raise ValueError(
                f"Unsupported appearance_mode='{appearance_mode}'. "
                f"Supported modes: {sorted(valid_modes)}"
            )

        self.appearance_mode = appearance_mode

        self.patch_ratio = float(patch_ratio)
        self.min_keypoint_conf = float(min_keypoint_conf)

        self.use_keypoint_confidence = bool(
            use_keypoint_confidence
        )

        self.use_region_confidence = bool(
            use_region_confidence
        )

        self.region_dilation_ratio = float(
            region_dilation_ratio
        )

    # ============================================================
    # Public API
    # ============================================================

    def extract(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray,
    ):
        """
        Extract appearance according to appearance_mode.

        Parameters
        ----------
        frame:
            BGR image.

        bbox:
            [x1, y1, x2, y2].

        keypoints:
            Array with shape:
                [N, 3]
            where each keypoint is:
                [x, y, confidence]

        Returns
        -------
        np.ndarray
            Appearance feature vector.

        For region modes, the return value is a concatenated
        region descriptor.
        """

        keypoints = self._prepare_keypoints(keypoints)

        if self.appearance_mode == "patch":
            return self._extract_patch_embedding(
                frame,
                bbox,
                keypoints,
            )

        if self.appearance_mode == "skeleton":
            return self._extract_skeleton_embedding(
                frame,
                bbox,
                keypoints,
            )

        if self.appearance_mode == "hybrid":
            patch_feature = self._extract_patch_embedding(
                frame,
                bbox,
                keypoints,
            )

            skeleton_feature = self._extract_skeleton_embedding(
                frame,
                bbox,
                keypoints,
            )

            return self._normalize(
                np.concatenate(
                    [
                        patch_feature,
                        skeleton_feature,
                    ]
                )
            )

        if self.appearance_mode == "region":
            return self._extract_region_embedding(
                frame,
                bbox,
                keypoints,
            )

        if self.appearance_mode == "region_hybrid":
            region_feature = self._extract_region_embedding(
                frame,
                bbox,
                keypoints,
            )

            skeleton_feature = self._extract_skeleton_embedding(
                frame,
                bbox,
                keypoints,
            )

            return self._normalize(
                np.concatenate(
                    [
                        region_feature,
                        skeleton_feature,
                    ]
                )
            )

        raise RuntimeError(
            f"Unsupported appearance mode: "
            f"{self.appearance_mode}"
        )

    def extract_batch(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        keypoints_batch: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Extract appearance features for multiple detections.
        """

        features = []

        for bbox, keypoints in zip(
            boxes,
            keypoints_batch,
        ):
            feature = self.extract(
                frame,
                bbox,
                keypoints,
            )

            features.append(feature)

        return features

    def extract_region_features(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray,
    ) -> Tuple[
        Dict[str, Optional[np.ndarray]],
        Dict[str, float],
    ]:
        """
        Extract independent appearance features for:

            head
            torso
            legs

        Returns
        -------
        region_features:
            {
                "head": feature or None,
                "torso": feature or None,
                "legs": feature or None,
            }

        region_confidences:
            {
                "head": confidence,
                "torso": confidence,
                "legs": confidence,
            }

        Missing regions are returned as None.
        """

        keypoints = self._prepare_keypoints(keypoints)

        region_features = {}
        region_confidences = {}

        for region_name in self.REGION_NAMES:

            feature, confidence = self._extract_single_region(
                frame=frame,
                bbox=bbox,
                keypoints=keypoints,
                region_name=region_name,
            )

            region_features[region_name] = feature
            region_confidences[region_name] = confidence

        return (
            region_features,
            region_confidences,
        )

    @staticmethod
    def cosine_distance(
        feature_a: Optional[np.ndarray],
        feature_b: Optional[np.ndarray],
    ) -> float:
        """
        Compute cosine distance.

        Returns 1.0 if either feature is unavailable.
        """

        if feature_a is None or feature_b is None:
            return 1.0

        feature_a = np.asarray(
            feature_a,
            dtype=np.float32,
        )

        feature_b = np.asarray(
            feature_b,
            dtype=np.float32,
        )

        norm_a = np.linalg.norm(feature_a)
        norm_b = np.linalg.norm(feature_b)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 1.0

        similarity = float(
            np.dot(feature_a, feature_b)
            / (norm_a * norm_b)
        )

        similarity = np.clip(
            similarity,
            -1.0,
            1.0,
        )

        return float(1.0 - similarity)

    # ============================================================
    # Keypoint processing
    # ============================================================

    def _prepare_keypoints(
        self,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """
        Convert keypoints into a consistent float32 Nx3 format.
        """

        if keypoints is None:
            return np.zeros(
                (17, 3),
                dtype=np.float32,
            )

        keypoints = np.asarray(
            keypoints,
            dtype=np.float32,
        )

        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(
                -1,
                3,
            )

        if keypoints.shape[1] < 3:
            raise ValueError(
                "Keypoints must have shape [N, 3]."
            )

        return keypoints[:, :3]

    def _valid_keypoint(
        self,
        keypoints: np.ndarray,
        index: int,
    ) -> bool:
        """
        Check whether a keypoint is valid.
        """

        if index >= len(keypoints):
            return False

        x, y, confidence = keypoints[index]

        if confidence < self.min_keypoint_conf:
            return False

        if not np.isfinite(x):
            return False

        if not np.isfinite(y):
            return False

        return True

    # ============================================================
    # Local descriptor
    # ============================================================

    def _extract_local_feature(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract local appearance descriptor.

        Descriptor:
            HSV histogram
            +
            RGB mean/std

        Total dimension:
            16 + 8 + 8 + 6 = 38
        """

        if frame is None or mask is None:
            return None

        if mask.size == 0:
            return None

        valid_pixels = mask > 0

        if not np.any(valid_pixels):
            return None

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        hsv_pixels = hsv[valid_pixels]
        rgb_pixels = rgb[valid_pixels]

        if len(hsv_pixels) == 0:
            return None

        # --------------------------------------------------------
        # HSV histogram
        # --------------------------------------------------------

        hist_h = cv2.calcHist(
            [hsv],
            [0],
            mask,
            [self.HSV_H_BINS],
            [0, 180],
        )

        hist_s = cv2.calcHist(
            [hsv],
            [1],
            mask,
            [self.HSV_S_BINS],
            [0, 256],
        )

        hist_v = cv2.calcHist(
            [hsv],
            [2],
            mask,
            [self.HSV_V_BINS],
            [0, 256],
        )

        hist_h = hist_h.flatten()
        hist_s = hist_s.flatten()
        hist_v = hist_v.flatten()

        hist = np.concatenate(
            [
                hist_h,
                hist_s,
                hist_v,
            ]
        ).astype(np.float32)

        hist_norm = np.linalg.norm(hist)

        if hist_norm > 1e-12:
            hist /= hist_norm

        # --------------------------------------------------------
        # RGB statistics
        # --------------------------------------------------------

        rgb_mean = np.mean(
            rgb_pixels,
            axis=0,
        )

        rgb_std = np.std(
            rgb_pixels,
            axis=0,
        )

        rgb_mean /= 255.0
        rgb_std /= 255.0

        rgb_stats = np.concatenate(
            [
                rgb_mean,
                rgb_std,
            ]
        ).astype(np.float32)

        feature = np.concatenate(
            [
                hist,
                rgb_stats,
            ]
        )

        return self._normalize(feature)

    # ============================================================
    # Patch appearance
    # ============================================================

    def _extract_patch_embedding(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """
        Extract local appearance around each selected keypoint.

        Each keypoint produces a 38-D local descriptor.

        11 selected keypoints:

            11 x 38 = 418-D

        Keypoint confidence weighting is controlled by:
            use_keypoint_confidence
        """

        bbox = self._clip_bbox(
            bbox,
            frame.shape,
        )

        x1, y1, x2, y2 = bbox

        bbox_w = max(
            1,
            x2 - x1,
        )

        bbox_h = max(
            1,
            y2 - y1,
        )

        patch_size = max(
            2,
            int(
                self.patch_ratio
                * max(
                    bbox_w,
                    bbox_h,
                )
            ),
        )

        features = []

        for kp_index in self.SELECTED_KEYPOINTS:

            if not self._valid_keypoint(
                keypoints,
                kp_index,
            ):
                features.append(
                    np.zeros(
                        self.LOCAL_FEATURE_DIM,
                        dtype=np.float32,
                    )
                )
                continue

            x, y, confidence = keypoints[kp_index]

            x = int(round(x))
            y = int(round(y))

            mask = np.zeros(
                frame.shape[:2],
                dtype=np.uint8,
            )

            cv2.circle(
                mask,
                (x, y),
                patch_size,
                255,
                -1,
            )

            # Keep the patch inside the person bounding box.
            bbox_mask = np.zeros_like(mask)

            cv2.rectangle(
                bbox_mask,
                (x1, y1),
                (x2, y2),
                255,
                -1,
            )

            mask = cv2.bitwise_and(
                mask,
                bbox_mask,
            )

            feature = self._extract_local_feature(
                frame,
                mask,
            )

            if feature is None:
                feature = np.zeros(
                    self.LOCAL_FEATURE_DIM,
                    dtype=np.float32,
                )

            if self.use_keypoint_confidence:
                feature = feature * float(
                    confidence
                )

            features.append(feature)

        embedding = np.concatenate(
            features
        ).astype(np.float32)

        return self._normalize(embedding)

    # ============================================================
    # Skeleton appearance
    # ============================================================

    def _extract_skeleton_embedding(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """
        Extract appearance from the complete visible skeleton.
        """

        mask = self._build_skeleton_mask(
            frame.shape[:2],
            bbox,
            keypoints,
        )

        feature = self._extract_local_feature(
            frame,
            mask,
        )

        if feature is None:
            return np.zeros(
                self.LOCAL_FEATURE_DIM,
                dtype=np.float32,
            )

        return self._normalize(feature)

    def _build_skeleton_mask(
        self,
        image_shape: Tuple[int, int],
        bbox: np.ndarray,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """
        Build skeleton mask.
        """

        height, width = image_shape

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        bbox = self._clip_bbox(
            bbox,
            (height, width, 3),
        )

        x1, y1, x2, y2 = bbox

        # Draw skeleton connections.
        for p1, p2 in self.SKELETON_CONNECTIONS:

            if not self._valid_keypoint(
                keypoints,
                p1,
            ):
                continue

            if not self._valid_keypoint(
                keypoints,
                p2,
            ):
                continue

            x_a, y_a, _ = keypoints[p1]
            x_b, y_b, _ = keypoints[p2]

            cv2.line(
                mask,
                (
                    int(round(x_a)),
                    int(round(y_a)),
                ),
                (
                    int(round(x_b)),
                    int(round(y_b)),
                ),
                255,
                8,
            )

        # Draw joints.
        for index in self.SELECTED_KEYPOINTS:

            if not self._valid_keypoint(
                keypoints,
                index,
            ):
                continue

            x, y, _ = keypoints[index]

            cv2.circle(
                mask,
                (
                    int(round(x)),
                    int(round(y)),
                ),
                8,
                255,
                -1,
            )

        # Restrict to bbox.
        bbox_mask = np.zeros_like(mask)

        cv2.rectangle(
            bbox_mask,
            (x1, y1),
            (x2, y2),
            255,
            -1,
        )

        mask = cv2.bitwise_and(
            mask,
            bbox_mask,
        )

        # Small dilation to create a local body corridor.
        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1,
        )

        return mask

    # ============================================================
    # Region appearance
    # ============================================================

    def _extract_region_embedding(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """
        Extract concatenated anatomical region descriptor.

        Region order:
            head
            torso
            legs

        Dimension:
            3 x 38 = 114-D
        """

        region_features, _ = self.extract_region_features(
            frame,
            bbox,
            keypoints,
        )

        features = []

        for region_name in self.REGION_NAMES:

            feature = region_features[
                region_name
            ]

            if feature is None:
                feature = np.zeros(
                    self.LOCAL_FEATURE_DIM,
                    dtype=np.float32,
                )

            features.append(feature)

        embedding = np.concatenate(
            features
        ).astype(np.float32)

        return self._normalize(embedding)

    def _extract_single_region(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray,
        region_name: str,
    ) -> Tuple[
        Optional[np.ndarray],
        float,
    ]:
        """
        Extract one anatomical region.

        Returns
        -------
        feature:
            Region appearance descriptor or None.

        confidence:
            Mean confidence of valid region keypoints.
        """

        if region_name not in self.REGION_NAMES:
            raise ValueError(
                f"Unknown region: {region_name}"
            )

        region_indices = self.REGION_KEYPOINTS[
            region_name
        ]

        valid_indices = []

        for index in region_indices:

            if self._valid_keypoint(
                keypoints,
                index,
            ):
                valid_indices.append(index)

        if len(valid_indices) == 0:
            return None, 0.0

        confidence_values = [
            float(keypoints[index][2])
            for index in valid_indices
        ]

        region_confidence = float(
            np.mean(confidence_values)
        )

        mask = self._build_region_mask(
            frame.shape[:2],
            bbox,
            keypoints,
            region_name,
        )

        feature = self._extract_local_feature(
            frame,
            mask,
        )

        if feature is None:
            return None, region_confidence

        if self.use_region_confidence:
            feature = feature * region_confidence

        feature = self._normalize(feature)

        return feature, region_confidence

    def _build_region_mask(
        self,
        image_shape: Tuple[int, int],
        bbox: np.ndarray,
        keypoints: np.ndarray,
        region_name: str,
    ) -> np.ndarray:
        """
        Build an anatomical region mask.

        Head:
            Convex hull of head keypoints.

        Torso:
            Convex hull of shoulders and hips.

        Legs:
            Limb corridors instead of one large polygon.
            This avoids excessive background between the legs.
        """

        height, width = image_shape

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        region_indices = self.REGION_KEYPOINTS[
            region_name
        ]

        valid_points = []

        for index in region_indices:

            if not self._valid_keypoint(
                keypoints,
                index,
            ):
                continue

            x, y, _ = keypoints[index]

            valid_points.append(
                [
                    int(round(x)),
                    int(round(y)),
                ]
            )

        if len(valid_points) == 0:
            return mask

        points = np.asarray(
            valid_points,
            dtype=np.int32,
        )

        # --------------------------------------------------------
        # Head
        # --------------------------------------------------------

        if region_name == "head":

            if len(points) >= 3:

                hull = cv2.convexHull(
                    points
                )

                cv2.fillConvexPoly(
                    mask,
                    hull,
                    255,
                )

            else:

                for point in points:

                    cv2.circle(
                        mask,
                        tuple(point),
                        12,
                        255,
                        -1,
                    )

        # --------------------------------------------------------
        # Torso
        # --------------------------------------------------------

        elif region_name == "torso":

            if len(points) >= 3:

                hull = cv2.convexHull(
                    points
                )

                cv2.fillConvexPoly(
                    mask,
                    hull,
                    255,
                )

            else:

                for point in points:

                    cv2.circle(
                        mask,
                        tuple(point),
                        15,
                        255,
                        -1,
                    )

        # --------------------------------------------------------
        # Legs
        # --------------------------------------------------------

        elif region_name == "legs":

            for p1, p2 in self.REGION_CONNECTIONS[
                "legs"
            ]:

                if not self._valid_keypoint(
                    keypoints,
                    p1,
                ):
                    continue

                if not self._valid_keypoint(
                    keypoints,
                    p2,
                ):
                    continue

                x1, y1, _ = keypoints[p1]
                x2, y2, _ = keypoints[p2]

                cv2.line(
                    mask,
                    (
                        int(round(x1)),
                        int(round(y1)),
                    ),
                    (
                        int(round(x2)),
                        int(round(y2)),
                    ),
                    255,
                    14,
                )

        # --------------------------------------------------------
        # Dilation
        # --------------------------------------------------------

        bbox = self._clip_bbox(
            bbox,
            (height, width, 3),
        )

        x1, y1, x2, y2 = bbox

        bbox_mask = np.zeros_like(mask)

        cv2.rectangle(
            bbox_mask,
            (x1, y1),
            (x2, y2),
            255,
            -1,
        )

        mask = cv2.bitwise_and(
            mask,
            bbox_mask,
        )

        bbox_w = max(
            1,
            x2 - x1,
        )

        bbox_h = max(
            1,
            y2 - y1,
        )

        dilation_size = max(
            1,
            int(
                self.region_dilation_ratio
                * max(
                    bbox_w,
                    bbox_h,
                )
            ),
        )

        kernel_size = (
            2 * dilation_size
        ) + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1,
        )

        mask = cv2.bitwise_and(
            mask,
            bbox_mask,
        )

        return mask

    # ============================================================
    # Utility
    # ============================================================

    @staticmethod
    def _clip_bbox(
        bbox: np.ndarray,
        image_shape,
    ) -> Tuple[int, int, int, int]:
        """
        Clip bbox to image boundaries.
        """

        height = image_shape[0]
        width = image_shape[1]

        bbox = np.asarray(
            bbox,
            dtype=np.float32,
        ).reshape(-1)

        if len(bbox) < 4:
            raise ValueError(
                "Bounding box must contain "
                "[x1, y1, x2, y2]."
            )

        x1, y1, x2, y2 = bbox[:4]

        x1 = int(
            np.clip(
                round(x1),
                0,
                width - 1,
            )
        )

        y1 = int(
            np.clip(
                round(y1),
                0,
                height - 1,
            )
        )

        x2 = int(
            np.clip(
                round(x2),
                x1 + 1,
                width - 1,
            )
        )

        y2 = int(
            np.clip(
                round(y2),
                y1 + 1,
                height - 1,
            )
        )

        return x1, y1, x2, y2

    @staticmethod
    def _normalize(
        feature: np.ndarray,
    ) -> np.ndarray:
        """
        L2 normalize feature vector.
        """

        feature = np.asarray(
            feature,
            dtype=np.float32,
        )

        norm = np.linalg.norm(
            feature
        )

        if norm < 1e-12:
            return feature

        return feature / norm

    # ============================================================
    # Dimension helpers
    # ============================================================

    def feature_dimension(self) -> int:
        """
        Return feature dimension for current mode.
        """

        if self.appearance_mode == "patch":
            return (
                len(self.SELECTED_KEYPOINTS)
                * self.LOCAL_FEATURE_DIM
            )

        if self.appearance_mode == "skeleton":
            return self.LOCAL_FEATURE_DIM

        if self.appearance_mode == "hybrid":
            return (
                len(self.SELECTED_KEYPOINTS)
                * self.LOCAL_FEATURE_DIM
                + self.LOCAL_FEATURE_DIM
            )

        if self.appearance_mode == "region":
            return (
                len(self.REGION_NAMES)
                * self.LOCAL_FEATURE_DIM
            )

        if self.appearance_mode == "region_hybrid":
            return (
                len(self.REGION_NAMES)
                * self.LOCAL_FEATURE_DIM
                + self.LOCAL_FEATURE_DIM
            )

        raise RuntimeError(
            f"Unknown appearance mode: "
            f"{self.appearance_mode}"
        )