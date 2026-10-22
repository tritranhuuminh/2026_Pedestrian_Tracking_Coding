import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

# ============================================================
# REMOVED
# ============================================================
# The original implementation used:
#
# from cython_bbox import bbox_overlaps as bbox_ious
#
# This dependency has been removed because cython_bbox requires
# compiling a C/C++ extension on Windows.
#
# A NumPy-based implementation is provided below instead.
# The input/output interface remains compatible with the
# original bbox_overlaps function.
# ============================================================

from tracker import kalman_filter


# ============================================================
# NumPy replacement for cython_bbox.bbox_overlaps
# ============================================================

def bbox_ious(atlbrs, btlbrs):
    """
    Compute pairwise Intersection over Union (IoU).

    This function replaces:
        cython_bbox.bbox_overlaps

    Parameters
    ----------
    atlbrs : np.ndarray
        First set of bounding boxes.
        Shape: (N, 4)

    btlbrs : np.ndarray
        Second set of bounding boxes.
        Shape: (M, 4)

    Bounding box format
    -------------------
    [x1, y1, x2, y2]

    Returns
    -------
    np.ndarray
        Pairwise IoU matrix with shape (N, M).
    """

    # Convert input arrays to float64.
    atlbrs = np.asarray(
        atlbrs,
        dtype=np.float64
    )

    btlbrs = np.asarray(
        btlbrs,
        dtype=np.float64
    )

    # Handle empty inputs.
    if atlbrs.size == 0 or btlbrs.size == 0:
        return np.zeros(
            (len(atlbrs), len(btlbrs)),
            dtype=np.float64
        )

    # Ensure the expected shape:
    #     (N, 4)
    #     (M, 4)
    atlbrs = atlbrs.reshape(-1, 4)
    btlbrs = btlbrs.reshape(-1, 4)

    # --------------------------------------------------------
    # Compute area of the first set of boxes.
    # --------------------------------------------------------

    a_width = np.maximum(
        0.0,
        atlbrs[:, 2] - atlbrs[:, 0]
    )

    a_height = np.maximum(
        0.0,
        atlbrs[:, 3] - atlbrs[:, 1]
    )

    area_a = a_width * a_height

    # --------------------------------------------------------
    # Compute area of the second set of boxes.
    # --------------------------------------------------------

    b_width = np.maximum(
        0.0,
        btlbrs[:, 2] - btlbrs[:, 0]
    )

    b_height = np.maximum(
        0.0,
        btlbrs[:, 3] - btlbrs[:, 1]
    )

    area_b = b_width * b_height

    # --------------------------------------------------------
    # Compute intersection coordinates.
    # --------------------------------------------------------

    xx1 = np.maximum(
        atlbrs[:, None, 0],
        btlbrs[None, :, 0]
    )

    yy1 = np.maximum(
        atlbrs[:, None, 1],
        btlbrs[None, :, 1]
    )

    xx2 = np.minimum(
        atlbrs[:, None, 2],
        btlbrs[None, :, 2]
    )

    yy2 = np.minimum(
        atlbrs[:, None, 3],
        btlbrs[None, :, 3]
    )

    # --------------------------------------------------------
    # Compute intersection width and height.
    # --------------------------------------------------------

    intersection_width = np.maximum(
        0.0,
        xx2 - xx1
    )

    intersection_height = np.maximum(
        0.0,
        yy2 - yy1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    # --------------------------------------------------------
    # Compute union area.
    # --------------------------------------------------------

    union = (
        area_a[:, None]
        + area_b[None, :]
        - intersection
    )

    # --------------------------------------------------------
    # Compute IoU.
    #
    # np.divide(..., where=...) prevents division by zero.
    # --------------------------------------------------------

    ious = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

    return ious


# ============================================================
# Match utilities
# ============================================================

def merge_matches(m1, m2, shape):

    O, P, Q = shape

    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix(
        (
            np.ones(len(m1)),
            (m1[:, 0], m1[:, 1])
        ),
        shape=(O, P)
    )

    M2 = scipy.sparse.coo_matrix(
        (
            np.ones(len(m2)),
            (m2[:, 0], m2[:, 1])
        ),
        shape=(P, Q)
    )

    mask = M1 * M2

    match = mask.nonzero()

    match = list(
        zip(
            match[0],
            match[1]
        )
    )

    unmatched_O = tuple(
        set(range(O))
        - set(
            [i for i, j in match]
        )
    )

    unmatched_Q = tuple(
        set(range(Q))
        - set(
            [j for i, j in match]
        )
    )

    return (
        match,
        unmatched_O,
        unmatched_Q
    )


def _indices_to_matches(
    cost_matrix,
    indices,
    thresh
):

    matched_cost = cost_matrix[
        tuple(zip(*indices))
    ]

    matched_mask = (
        matched_cost <= thresh
    )

    matches = indices[
        matched_mask
    ]

    unmatched_a = tuple(
        set(range(cost_matrix.shape[0]))
        - set(matches[:, 0])
    )

    unmatched_b = tuple(
        set(range(cost_matrix.shape[1]))
        - set(matches[:, 1])
    )

    return (
        matches,
        unmatched_a,
        unmatched_b
    )


# ============================================================
# Linear assignment
# ============================================================

def linear_assignment(
    cost_matrix,
    thresh
):

    if cost_matrix.size == 0:

        return (
            np.empty(
                (0, 2),
                dtype=int
            ),
            tuple(
                range(
                    cost_matrix.shape[0]
                )
            ),
            tuple(
                range(
                    cost_matrix.shape[1]
                )
            )
        )

    matches = []
    unmatched_a = []
    unmatched_b = []

    # LAPJV solves the linear assignment problem.
    cost, x, y = lap.lapjv(
        cost_matrix,
        extend_cost=True,
        cost_limit=thresh
    )

    for ix, mx in enumerate(x):

        if mx >= 0:
            matches.append(
                [ix, mx]
            )

    unmatched_a = np.where(
        x < 0
    )[0]

    unmatched_b = np.where(
        y < 0
    )[0]

    matches = np.asarray(
        matches
    )

    return (
        matches,
        unmatched_a,
        unmatched_b
    )


# ============================================================
# IoU utilities
# ============================================================

def ious(atlbrs, btlbrs):
    """
    Compute pairwise IoU.

    Parameters
    ----------
    atlbrs : list or np.ndarray
        First set of bounding boxes.

    btlbrs : list or np.ndarray
        Second set of bounding boxes.

    Returns
    -------
    np.ndarray
        IoU matrix with shape:
        (len(atlbrs), len(btlbrs))
    """

    # --------------------------------------------------------
    # CHANGED
    # --------------------------------------------------------
    # Original implementation:
    #
    # ious = np.zeros(
    #     (len(atlbrs), len(btlbrs)),
    #     dtype=np.float
    # )
    #
    # if ious.size == 0:
    #     return ious
    #
    # ious = bbox_ious(
    #     np.ascontiguousarray(
    #         atlbrs,
    #         dtype=np.float
    #     ),
    #     np.ascontiguousarray(
    #         btlbrs,
    #         dtype=np.float
    #     )
    # )
    #
    # return ious
    #
    # New implementation:
    # Use the NumPy-based bbox_ious() defined above.
    #
    # np.float was also replaced with np.float64 because
    # np.float was removed in NumPy 2.x.
    # --------------------------------------------------------

    atlbrs = np.asarray(
        atlbrs,
        dtype=np.float64
    )

    btlbrs = np.asarray(
        btlbrs,
        dtype=np.float64
    )

    return bbox_ious(
        atlbrs,
        btlbrs
    )


def tlbr_expand(
    tlbr,
    scale=1.2
):

    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(
    atracks,
    btracks
):
    """
    Compute IoU distance between tracks and detections.

    The returned value is:

        distance = 1 - IoU

    Parameters
    ----------
    atracks : list[STrack] or np.ndarray
    btracks : list[STrack] or np.ndarray

    Returns
    -------
    np.ndarray
        IoU distance matrix.
    """

    if (
        len(atracks) > 0
        and isinstance(
            atracks[0],
            np.ndarray
        )
    ) or (
        len(btracks) > 0
        and isinstance(
            btracks[0],
            np.ndarray
        )
    ):

        atlbrs = atracks
        btlbrs = btracks

    else:

        atlbrs = [
            track.tlbr
            for track in atracks
        ]

        btlbrs = [
            track.tlbr
            for track in btracks
        ]

    _ious = ious(
        atlbrs,
        btlbrs
    )

    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(
    atracks,
    btracks
):
    """
    Compute IoU distance using predicted bounding boxes.

    Parameters
    ----------
    atracks : list[STrack]
    btracks : list[STrack]

    Returns
    -------
    np.ndarray
        IoU distance matrix.
    """

    if (
        len(atracks) > 0
        and isinstance(
            atracks[0],
            np.ndarray
        )
    ) or (
        len(btracks) > 0
        and isinstance(
            btracks[0],
            np.ndarray
        )
    ):

        atlbrs = atracks
        btlbrs = btracks

    else:

        atlbrs = [
            track.tlwh_to_tlbr(
                track.pred_bbox
            )
            for track in atracks
        ]

        btlbrs = [
            track.tlwh_to_tlbr(
                track.pred_bbox
            )
            for track in btracks
        ]

    _ious = ious(
        atlbrs,
        btlbrs
    )

    cost_matrix = 1 - _ious

    return cost_matrix


# ============================================================
# Appearance embedding distance
# ============================================================

def embedding_distance(
    tracks,
    detections,
    metric='cosine'
):
    """
    Compute appearance embedding distance.

    Parameters
    ----------
    tracks : list[STrack]
        Existing tracks.

    detections : list[STrack]
        Current detections.

    metric : str
        Distance metric used by scipy.spatial.distance.cdist.

    Returns
    -------
    np.ndarray
        Appearance distance matrix.
    """

    # --------------------------------------------------------
    # CHANGED
    # --------------------------------------------------------
    # np.float was replaced with np.float64 because np.float
    # was removed in NumPy 2.x.
    # --------------------------------------------------------

    cost_matrix = np.zeros(
        (
            len(tracks),
            len(detections)
        ),
        dtype=np.float64
    )

    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray(
        [
            track.curr_feat
            for track in detections
        ],
        dtype=np.float64
    )

    track_features = np.asarray(
        [
            track.smooth_feat
            for track in tracks
        ],
        dtype=np.float64
    )

    cost_matrix = np.maximum(
        0.0,
        cdist(
            track_features,
            det_features,
            metric
        )
    )

    return cost_matrix


# ============================================================
# Kalman filter gating
# ============================================================

def gate_cost_matrix(
    kf,
    cost_matrix,
    tracks,
    detections,
    only_position=False
):

    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = (
        2
        if only_position
        else 4
    )

    gating_threshold = (
        kalman_filter.chi2inv95[
            gating_dim
        ]
    )

    # Original XYAH measurement representation:
    #
    # measurements = np.asarray(
    #     [det.to_xyah() for det in detections]
    # )

    # Current implementation uses XYWH.
    measurements = np.asarray(
        [
            det.to_xywh()
            for det in detections
        ]
    )

    for row, track in enumerate(tracks):

        gating_distance = (
            kf.gating_distance(
                track.mean,
                track.covariance,
                measurements,
                only_position
            )
        )

        cost_matrix[
            row,
            gating_distance > gating_threshold
        ] = np.inf

    return cost_matrix


def fuse_motion(
    kf,
    cost_matrix,
    tracks,
    detections,
    only_position=False,
    lambda_=0.98
):

    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = (
        2
        if only_position
        else 4
    )

    gating_threshold = (
        kalman_filter.chi2inv95[
            gating_dim
        ]
    )

    # Original XYAH measurement representation:
    #
    # measurements = np.asarray(
    #     [det.to_xyah() for det in detections]
    # )

    # Current implementation uses XYWH.
    measurements = np.asarray(
        [
            det.to_xywh()
            for det in detections
        ]
    )

    for row, track in enumerate(tracks):

        gating_distance = (
            kf.gating_distance(
                track.mean,
                track.covariance,
                measurements,
                only_position,
                metric='maha'
            )
        )

        cost_matrix[
            row,
            gating_distance > gating_threshold
        ] = np.inf

        cost_matrix[row] = (
            lambda_ * cost_matrix[row]
            +
            (1 - lambda_) * gating_distance
        )

    return cost_matrix


# ============================================================
# Fuse ReID and IoU
# ============================================================

def fuse_iou(
    cost_matrix,
    tracks,
    detections
):

    if cost_matrix.size == 0:
        return cost_matrix

    reid_sim = 1 - cost_matrix

    iou_dist = iou_distance(
        tracks,
        detections
    )

    iou_sim = 1 - iou_dist

    fuse_sim = (
        reid_sim
        * (1 + iou_sim)
        / 2
    )

    det_scores = np.array(
        [
            det.score
            for det in detections
        ]
    )

    det_scores = np.expand_dims(
        det_scores,
        axis=0
    ).repeat(
        cost_matrix.shape[0],
        axis=0
    )

    # Optional detection-score fusion from the original code:
    #
    # fuse_sim = (
    #     fuse_sim
    #     * (1 + det_scores)
    #     / 2
    # )

    fuse_cost = 1 - fuse_sim

    return fuse_cost


# ============================================================
# Fuse IoU distance with detection confidence
# ============================================================

def fuse_score(
    cost_matrix,
    detections
):

    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1 - cost_matrix

    det_scores = np.array(
        [
            det.score
            for det in detections
        ]
    )

    det_scores = np.expand_dims(
        det_scores,
        axis=0
    ).repeat(
        cost_matrix.shape[0],
        axis=0
    )

    fuse_sim = (
        iou_sim
        * det_scores
    )

    fuse_cost = 1 - fuse_sim

    return fuse_cost

def max_ioa(tlbrs):
    """Tỉ lệ diện tích của mỗi box bị box khác phủ (max). Shape (N,)."""
    b = np.asarray(tlbrs, dtype=np.float64).reshape(-1, 4)
    if len(b) < 2:
        return np.zeros(len(b))
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = np.maximum(1e-6, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    ioa = inter / area[:, None]          # phần của box i bị box j phủ
    np.fill_diagonal(ioa, 0.0)
    return ioa.max(1)

def appearance_refine(ious_dists, ious_dists_mask, emb, valid, lam=0.3):

    """
    ious_dists      : cost IoU (đã fuse_score)
    ious_dists_mask : True = ngoài vùng proximity
    emb             : khoảng cách cosine thô (1 - cos), shape như ious_dists
    valid           : True nếu cả track và detection đều có feature
    """
    cand = (~ious_dists_mask) & valid
    cnt = cand.sum(0)                                   # số ứng viên mỗi detection
    multi = cand & (cnt >= 2)[None, :]
    if not multi.any():
        return ious_dists

    mean_c = np.where(cand, emb, 0.0).sum(0) / np.maximum(cnt, 1)
    centered = np.where(multi, emb - mean_c[None, :], 0.0)
    sd = centered[multi].std() + 1e-6

    delta = lam * np.tanh(centered / sd)                # bị chặn trong ±lam
    return np.clip(ious_dists + delta, 0.0, 1.0)

# ---------- mục 1+3: độ tương đồng theo part, có mask, gallery top-k ----------
def _part_sim_block(Fa, Wa, Fb, Wb):
    s = np.einsum('apk,bpk->abp', Fa, Fb)                    # (A,B,P)
    w = Wa[:, None, :] * Wb[None, :, :]
    den = w.sum(2)
    num = (w * s).sum(2)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)


def part_similarity_matrix(tracks, dets, topk=3, use_gallery=True):
    """(Nt,Nd) similarity, NaN nếu không so được."""
    Nt, Nd = len(tracks), len(dets)
    sim = np.full((Nt, Nd), np.nan)
    ref = next((d.curr_parts for d in dets if d.curr_parts is not None), None)
    if ref is None or Nt == 0 or Nd == 0:
        return sim
    Dp = np.stack([d.curr_parts if d.curr_parts is not None else np.zeros_like(ref) for d in dets])
    Dw = np.stack([d.curr_w if d.curr_w is not None else np.zeros(ref.shape[0], np.float32) for d in dets])

    for i, t in enumerate(tracks):
        entries = []
        if t.smooth_parts is not None:
            entries.append((t.smooth_parts, t.smooth_w))
        if use_gallery:
            entries += list(t.gallery)
        if not entries:
            continue
        Fp = np.stack([e[0] for e in entries])
        Fw = np.stack([e[1] for e in entries])
        s = _part_sim_block(Fp, Fw, Dp, Dw)                  # (E,Nd)
        s2 = np.where(np.isnan(s), -np.inf, s)
        top = np.sort(s2, axis=0)[::-1][:topk]
        fin = np.isfinite(top)
        cnt = fin.sum(0)
        tot = np.where(fin, top, 0.0).sum(0)
        sim[i] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return sim


# ---------- mục 2: pose bất biến theo bbox ----------
def pose_distance(tracks, dets, sigma=0.1, sl=slice(5, 17)):
    Nt, Nd = len(tracks), len(dets)
    out = np.full((Nt, Nd), np.nan)
    if Nt == 0 or Nd == 0:
        return out
    K = 12
    def pack(objs):
        k = np.stack([o.kp_norm[sl] if o.kp_norm is not None else np.zeros((K, 2), np.float32) for o in objs])
        c = np.stack([o.kp_conf[sl] if o.kp_conf is not None else np.zeros(K, np.float32) for o in objs])
        return k, c
    Tk, Tc = pack(tracks)
    Dk, Dc = pack(dets)
    d2 = ((Tk[:, None] - Dk[None]) ** 2).sum(-1)             # (Nt,Nd,K)
    w = np.minimum(Tc[:, None], Dc[None])
    den = w.sum(-1)
    num = (w * np.exp(-d2 / (2 * sigma ** 2))).sum(-1)
    return np.where(den > 1e-6, 1.0 - num / np.maximum(den, 1e-6), np.nan)


# ---------- fusion (thay hàm cũ) ----------
def _center_scale(cost, cand):
    cnt = cand.sum(0)
    multi = cand & (cnt >= 2)[None, :]
    if not multi.any():
        return np.zeros_like(cost), multi
    mean_c = np.where(cand, cost, 0.0).sum(0) / np.maximum(cnt, 1)
    c = np.where(multi, cost - mean_c[None, :], 0.0)
    return c / (c[multi].std() + 1e-6), multi


def appearance_refine(ious_dists, far_mask, cues, lam=0.3):
    """cues: list[(cost, valid, weight)]; far_mask True = không phải ứng viên."""
    near = ~far_mask
    total = np.zeros_like(ious_dists)
    wsum = np.zeros_like(ious_dists)
    for cost, valid, w in cues:
        z, m = _center_scale(cost, near & valid)
        total += w * z
        wsum += w * m
    z = np.where(wsum > 0, total / np.maximum(wsum, 1e-9), 0.0)
    return np.clip(ious_dists + lam * np.tanh(z), 0.0, 1.0)


# ---------- mục 4b: ứng viên cho track lost theo gating Kalman (chi2 95%) ----------
def kalman_gate_mask(kf, tracks, detections, rows, only_position=False):
    mask = np.zeros((len(tracks), len(detections)), bool)
    if mask.size == 0:
        return mask
    thr = kalman_filter.chi2inv95[2 if only_position else 4]
    meas = np.asarray([d.to_xywh() for d in detections])
    for i, t in enumerate(tracks):
        if not rows[i]:
            continue
        gd = kf.gating_distance(t.mean, t.covariance, meas, only_position, metric='maha')
        mask[i] = gd <= thr
    return mask