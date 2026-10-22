import cv2
import numpy as np


class PartAppearanceExtractor:
    """
    keypoints -> gate -> polygon (scale-aware) -> Lab hist / part
              -> per-part feature (P, D) + quality + valid
    Không fuse thành 1 vector; dùng pairwise_distance() để so khớp theo part.
    """

    PARTS = (
        ("torso", ((5, 6, 12, 11),)),
        ("thigh", ((11, 13), (12, 14))),
        ("shin",  ((13, 15), (14, 16))),
    )
    PART_W = np.array([1.0, 0.6, 0.4], np.float32)
    NUM_PARTS = 3
    BINS = 8
    DIM = BINS * BINS + BINS            # 72 / part
    AB_LO, AB_HI = 80, 176              # dải a,b thực tế (OpenCV uint8 Lab)

    # min_bbox_h=40.0,
    # pad_ratio=0.05,
    # min_keypoint_conf=0.35,
    # half_w_ratio=0.06,
    # limb_w_ratio=0.30,
    # min_limb_ratio=0.08,
    # torso_shrink=0.7,
    # min_part_area_ratio=0.008,
    # min_pixels=30,
    # good_pixels=100,
    # l_weight=0.25,
    # occ_power=2.0,
    # debug=False
    
    def __init__(self,
                 min_bbox_h, pad_ratio,
                 min_keypoint_conf,
                 half_w_ratio, limb_w_ratio, min_limb_ratio,
                 torso_shrink, min_part_area_ratio,
                 min_pixels, good_pixels,
                 l_weight, occ_power, debug):
        for k, v in list(locals().items()):
            if k != "self":
                setattr(self, k, v)
        self.dbg = []                     # [(n, part, poly_int_in_image_coords)]

    # ------------------------------------------------------------------
    @staticmethod
    def to_pixel_keypoints(kps, image_shape, normalized=None):
        kps = np.array(kps, np.float32)
        if kps.ndim == 2:
            kps = kps[None]
        if kps.shape[2] == 2:
            kps = np.concatenate(
                [kps, np.ones(kps.shape[:2] + (1,), np.float32)], axis=2)
        xy = kps[..., :2]
        if normalized is None:        # chỉ đoán trên keypoint có conf > 0
            m = np.isfinite(xy).all(-1) & (kps[..., 2] > 0)
            normalized = bool(m.any() and xy[m].min() >= -0.05
                              and xy[m].max() <= 1.05)
        if normalized:
            h, w = image_shape[:2]
            kps[..., 0] *= (w - 1)
            kps[..., 1] *= (h - 1)
        kps[..., 2] = np.clip(np.nan_to_num(kps[..., 2], nan=0.0), 0, 1)
        return kps

    # ------------------------------------------------------------------
    @staticmethod
    def _occlusion_ratio(bb, ctx=None):
        """
        Tỉ lệ diện tích bbox bb[n] bị che bởi box trong ctx đứng *trước*
        (y2 lớn hơn, strict => tự nó không tự che chính nó).
        ctx=None -> dùng chính bb.
        """
        ctx = bb if ctx is None else ctx
        if len(bb) == 0 or len(ctx) == 0:
            return np.zeros(len(bb), np.float32)
        x1 = np.maximum(bb[:, None, 0], ctx[None, :, 0])
        y1 = np.maximum(bb[:, None, 1], ctx[None, :, 1])
        x2 = np.minimum(bb[:, None, 2], ctx[None, :, 2])
        y2 = np.minimum(bb[:, None, 3], ctx[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1]) + 1e-9
        ov = inter / area[:, None]
        front = ctx[None, :, 3] > bb[:, None, 3] + 1e-6
        return (ov * front).max(1).astype(np.float32)

    # ------------------------------------------------------------------
    def extract(self, image, kps, bboxes, ctx_boxes=None):
        """
        image   : BGR uint8
        kps     : (N,17,3) pixel coords (qua to_pixel_keypoints)
        bboxes  : (N,4+) xyxy, cùng thứ tự với kps
        ctx_boxes : (M,4) mọi box trong frame (kể cả low-score) để tính che khuất

        Returns feats (N,P,D), qual (N,P), valid (N,P), occ (N,)
        """
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
        N = len(kps)
        assert len(bboxes) == N, "kps và bboxes lệch số lượng"
        P, D = self.NUM_PARTS, self.DIM
        feats = np.zeros((N, P, D), np.float32)
        qual = np.zeros((N, P), np.float32)
        valid = np.zeros((N, P), bool)
        self.dbg = []
        if N == 0:
            return feats, qual, valid, np.zeros(0, np.float32)

        H, W = image.shape[:2]
        bb = np.asarray(bboxes, np.float32)[:, :4]
        ctx = None if ctx_boxes is None else np.asarray(ctx_boxes, np.float32)[:, :4]
        occ = self._occlusion_ratio(bb, ctx)

        for n in range(N):
            if not np.isfinite(bb[n]).all():
                continue
            x1, y1, x2, y2 = bb[n]
            bh = y2 - y1
            if bh < self.min_bbox_h:
                continue
            pad = self.pad_ratio * bh
            cx1, cy1 = int(max(0, np.floor(x1 - pad))), int(max(0, np.floor(y1 - pad)))
            cx2, cy2 = int(min(W, np.ceil(x2 + pad))), int(min(H, np.ceil(y2 + pad)))
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                continue

            lab = cv2.cvtColor(image[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2LAB)
            lab[..., 1:] = np.clip(lab[..., 1:], self.AB_LO, self.AB_HI - 1)

            kp = kps[n].copy()
            kp[:, 0] -= cx1
            kp[:, 1] -= cy1
            occ_f = (1.0 - occ[n]) ** self.occ_power

            for p, (_, groups) in enumerate(self.PARTS):
                mask = np.zeros(lab.shape[:2], np.uint8)
                confs = []
                for idx in groups:
                    idx = list(idx)
                    xy, c = kp[idx, :2], float(kp[idx, 2].min())
                    if c < self.min_keypoint_conf or not np.isfinite(xy).all():
                        continue
                    poly = self._polygon(xy, bh)
                    if poly is None:
                        continue
                    pts = np.clip(poly, -1e4, 1e4).round().astype(np.int32)
                    cv2.fillConvexPoly(mask, pts, 1)
                    confs.append(c)
                    if self.debug:
                        self.dbg.append((n, p, pts + np.array([cx1, cy1], np.int32)))
                if not confs:
                    continue
                res = self._hist(lab, mask)
                if res is None:
                    continue
                f, npix = res
                feats[n, p] = f
                valid[n, p] = True
                pix_q = min(1.0, npix / self.good_pixels)
                qual[n, p] = np.clip(np.mean(confs) * pix_q * occ_f, 0, 1)
        return feats, qual, valid, occ

    # ------------------------------------------------------------------
    def _polygon(self, xy, bh):
        if len(xy) == 4:                              # torso
            c = xy.mean(0)
            pts = (c + self.torso_shrink * (xy - c)).astype(np.float32)
            hull = cv2.convexHull(pts).reshape(-1, 2)
            if cv2.contourArea(hull) < self.min_part_area_ratio * bh * bh:
                return None
            return hull
        a, b = xy                                     # limb
        d = b - a
        L = float(np.linalg.norm(d))
        if L < self.min_limb_ratio * bh:
            return None
        half_w = min(self.limb_w_ratio * L, self.half_w_ratio * bh)
        nrm = np.array([-d[1], d[0]]) / L * half_w
        a2, b2 = a + 0.2 * d, a + 0.8 * d
        return np.stack([a2 + nrm, b2 + nrm, b2 - nrm, a2 - nrm])

    def _hist(self, lab, mask):
        npix = int(cv2.countNonZero(mask))
        if npix < self.min_pixels:
            return None
        B, lo, hi = self.BINS, self.AB_LO, self.AB_HI
        ab = cv2.calcHist([lab], [1, 2], mask, [B, B], [lo, hi, lo, hi])
        ab = cv2.GaussianBlur(ab.astype(np.float32), (3, 3), 0.8).ravel()
        l_ = cv2.calcHist([lab], [0], mask, [B], [0, 256]).ravel()
        if ab.sum() <= 0 or l_.sum() <= 0:
            return None
        f = np.concatenate([
            np.sqrt(ab / ab.sum()) * np.sqrt(1 - self.l_weight),
            np.sqrt(l_ / l_.sum()) * np.sqrt(self.l_weight),
        ]).astype(np.float32)
        f /= (np.linalg.norm(f) + 1e-9)
        return f, npix

    # ------------------------------------------------------------------
    @classmethod
    def pairwise_distance(cls, Ft, Vt, Fd, Vd, min_shared=0.3, calib=None,
                          return_mask=False):
        """
        Ft (T,P,D), Vt (T,P) : track       Fd (N,P,D), Vd (N,P) : detection
        -> d (T,N) in [0,1]; thiếu part chung -> 1.0 (không NaN).
        known (T,N) bool: cặp có đủ part chung để so sánh.
        calib=(d_same, d_diff): kéo giãn distance theo thống kê thực tế.
        """
        w = cls.PART_W
        sim = np.einsum('tpd,npd->tnp', Ft, Fd)
        bw = (Vt[:, None, :] & Vd[None, :, :]) * w[None, None, :]
        wsum = bw.sum(-1)
        d = 1.0 - (sim * bw).sum(-1) / np.maximum(wsum, 1e-9)
        if calib is not None:
            d = (d - calib[0]) / max(calib[1] - calib[0], 1e-6)
        d = np.clip(d, 0.0, 1.0)
        known = wsum >= min_shared * w.sum()
        d[~known] = 1.0
        return (d, known) if return_mask else d


def draw_debug(image, ext, qual, valid):
    """Gọi sau extract() với ext.debug=True để kiểm tra polygon bằng mắt."""
    vis = image.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for n, p, pts in ext.dbg:
        cv2.polylines(vis, [pts], True, colors[p], 1)
        if valid[n, p]:
            cv2.putText(vis, f"{qual[n, p]:.2f}", tuple(int(v) for v in pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[p], 1)
    return vis