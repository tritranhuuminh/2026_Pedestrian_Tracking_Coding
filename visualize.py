"""
visualize.py
============

Visualization utilities for YOLO26n-Pose + BoT-SORT.

Responsibilities
----------------
1. Draw BoT-SORT track bounding boxes.
2. Draw track IDs using the same yellow style as the bounding box.
3. Draw frame-level processing information.
4. Display current FPS and average FPS.

Visualization
-------------
BoT-SORT Track:
    Yellow bounding box
    ID <track_id>

IMPORTANT
---------
Only ONE bounding box is drawn for each track.

The previous separate:
    - YOLO yellow detection bbox
    - BoT-SORT green tracking bbox

visualization is removed.

The track ID is integrated into the yellow track visualization.
"""

import cv2
import numpy as np


# ============================================================
# Utility
# ============================================================

def _clip_box(frame, x1, y1, x2, y2):
    """
    Clip bounding-box coordinates to image boundaries.
    """

    h, w = frame.shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    return x1, y1, x2, y2


# ============================================================
# Track ID extraction
# ============================================================

def _get_track_id(track):
    """
    Robustly extract track ID from a BoT-SORT track object.

    Supports:
        track.track_id
        track.id

    Returns
    -------
    int or None
    """

    if track is None:
        return None

    # --------------------------------------------------------
    # Standard BoT-SORT
    # --------------------------------------------------------

    if hasattr(track, "track_id"):

        try:
            track_id = track.track_id

            if track_id is not None:
                return int(track_id)

        except (TypeError, ValueError):
            pass

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if hasattr(track, "id"):

        try:
            track_id = track.id

            if track_id is not None:
                return int(track_id)

        except (TypeError, ValueError):
            pass

    return None


# ============================================================
# Track bounding-box extraction
# ============================================================

def _get_track_tlbr(track):
    """
    Robustly extract current track bounding box.

    Preferred:
        track.tlbr

    Fallback:
        track.tlwh
    """

    if track is None:
        return None

    # --------------------------------------------------------
    # Standard tlbr
    # --------------------------------------------------------

    if hasattr(track, "tlbr"):

        try:

            tlbr = track.tlbr

            if hasattr(tlbr, "detach"):
                tlbr = tlbr.detach()

            if hasattr(tlbr, "cpu"):
                tlbr = tlbr.cpu()

            if hasattr(tlbr, "numpy"):
                tlbr = tlbr.numpy()

            values = np.asarray(
                tlbr,
                dtype=float,
            ).reshape(-1)

            if len(values) >= 4:
                return values[:4]

        except Exception:
            pass

    # --------------------------------------------------------
    # Fallback: tlwh
    # --------------------------------------------------------

    if hasattr(track, "tlwh"):

        try:

            tlwh = track.tlwh

            if hasattr(tlwh, "detach"):
                tlwh = tlwh.detach()

            if hasattr(tlwh, "cpu"):
                tlwh = tlwh.cpu()

            if hasattr(tlwh, "numpy"):
                tlwh = tlwh.numpy()

            values = np.asarray(
                tlwh,
                dtype=float,
            ).reshape(-1)

            if len(values) < 4:
                return None

            x, y, w, h = values[:4]

            return np.array(
                [
                    x,
                    y,
                    x + w,
                    y + h,
                ],
                dtype=float,
            )

        except Exception:
            pass

    return None


# ============================================================
# Track confidence extraction
# ============================================================

def _get_track_confidence(track):
    """
    Try to extract the current track confidence.

    This is optional and is NOT required for visualization.

    Supports common attributes:
        track.score
        track.conf
        track.confidence

    Returns
    -------
    float or None
    """

    if track is None:
        return None

    for attr in (
        "score",
        "conf",
        "confidence",
    ):

        if not hasattr(track, attr):
            continue

        try:

            value = getattr(track, attr)

            if value is None:
                continue

            confidence = float(value)

            if np.isfinite(confidence):
                return confidence

        except (TypeError, ValueError):
            continue

    return None


# ============================================================
# Track visualization
# ============================================================

def draw_tracks(frame, tracks):
    """
    Draw active BoT-SORT tracks.

    Visualization style
    --------------------
    Yellow bounding box
    Yellow label background
    Black ID text

    Example:
        ID 1
        ID 12

    IMPORTANT
    ---------
    Only the BoT-SORT track bounding box is drawn.

    No separate YOLO detection bounding box is drawn.
    No green bounding box is drawn.

    Every valid track with a valid track ID is displayed.
    """

    if frame is None:
        return frame

    if tracks is None:
        return frame

    try:
        track_list = list(tracks)
    except Exception:
        return frame

    for track in track_list:

        if track is None:
            continue

        # ----------------------------------------------------
        # Get bounding box
        # ----------------------------------------------------

        tlbr = _get_track_tlbr(track)

        if tlbr is None:
            continue

        try:

            x1, y1, x2, y2 = tlbr

            x1, y1, x2, y2 = _clip_box(
                frame,
                x1,
                y1,
                x2,
                y2,
            )

        except Exception:
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        # ----------------------------------------------------
        # Get track ID
        # ----------------------------------------------------

        track_id = _get_track_id(track)

        if track_id is None:
            # Do not draw anonymous tracks.
            continue

        # ====================================================
        # Yellow track bounding box
        # ====================================================

        YELLOW = (0, 255, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            YELLOW,
            2,
        )

        # ====================================================
        # ID label
        # ====================================================

        label = f"ID {track_id}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.60
        thickness = 2

        label_size, baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        label_w, label_h = label_size

        # ----------------------------------------------------
        # Label horizontal position
        # ----------------------------------------------------

        label_x1 = x1

        label_x2 = min(
            frame.shape[1] - 1,
            x1 + label_w + 10,
        )

        # ----------------------------------------------------
        # Prefer label ABOVE the bbox
        # ----------------------------------------------------

        label_y2 = y1

        label_y1 = (
            y1
            - label_h
            - baseline
            - 6
        )

        # ----------------------------------------------------
        # If bbox is near top of image,
        # put label INSIDE the bbox.
        # ----------------------------------------------------

        if label_y1 < 0:

            label_y1 = y1

            label_y2 = min(
                frame.shape[0] - 1,
                y1 + label_h + baseline + 6,
            )

        # ----------------------------------------------------
        # Make sure label remains valid
        # ----------------------------------------------------

        if label_y2 <= label_y1:
            continue

        # ====================================================
        # Yellow label background
        # ====================================================

        cv2.rectangle(
            frame,
            (label_x1, label_y1),
            (label_x2, label_y2),
            YELLOW,
            -1,
        )

        # ====================================================
        # Black ID text
        # ====================================================

        text_y = (
            label_y2
            - baseline
            - 2
        )

        cv2.putText(
            frame,
            label,
            (
                label_x1 + 5,
                text_y,
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# Frame information visualization
# ============================================================

def draw_frame_info(
    frame,
    frame_id,
    total_frames,
    num_tracks,
    num_detections,
    current_fps,
    average_fps,
):
    """
    Draw global processing information.

    Note
    ----
    num_detections is retained for API compatibility.

    Detection bounding boxes are NOT drawn.
    """

    if frame is None:
        return frame

    # --------------------------------------------------------
    # Background panel
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 185),
        (0, 0, 0),
        -1,
    )

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    if total_frames > 0:

        frame_text = (
            f"Frame: {frame_id}/{total_frames}"
        )

    else:

        frame_text = (
            f"Frame: {frame_id}"
        )

    cv2.putText(
        frame,
        frame_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # YOLO detections count
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Detections: {num_detections}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Active tracks
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Tracks: {num_tracks}",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Current FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {current_fps:.2f}",
        (20, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Average FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Avg FPS: {average_fps:.2f}",
        (20, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


# ============================================================
# Complete visualization pipeline
# ============================================================

def visualize_frame(
    frame,
    detections,
    tracks,
    frame_id,
    total_frames,
    current_fps,
    average_fps,
):
    """
    Apply complete visualization pipeline.

    Visualization order
    --------------------
    1. BoT-SORT tracks + IDs
    2. Frame information

    IMPORTANT
    ---------
    Detection bounding boxes are intentionally NOT drawn.

    The `detections` argument is retained for compatibility
    with the existing main.py call structure.
    """

    if frame is None:
        return frame

    if detections is None:
        detections = []

    if tracks is None:
        tracks = []

    # --------------------------------------------------------
    # 1. Draw BoT-SORT tracks + IDs
    # --------------------------------------------------------

    frame = draw_tracks(
        frame,
        tracks,
    )

    # --------------------------------------------------------
    # Track count
    # --------------------------------------------------------

    try:
        num_tracks = len(tracks)
    except Exception:
        num_tracks = 0

    # --------------------------------------------------------
    # Detection count
    #
    # Kept for frame-information display only.
    # No detection bbox is drawn.
    # --------------------------------------------------------

    try:
        num_detections = len(detections)
    except Exception:
        num_detections = 0

    # --------------------------------------------------------
    # 2. Draw frame information
    # --------------------------------------------------------

    frame = draw_frame_info(
        frame,
        frame_id,
        total_frames,
        num_tracks,
        num_detections,
        current_fps,
        average_fps,
    )

    return frame

