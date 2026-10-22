import time
from contextlib import contextmanager

import numpy as np
import torch

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from tracker.part_appearance import PartAppearanceExtractor


# ============================================================================
# Latency instrumentation
#   extract          : keypoint -> part features (PartAppearanceExtractor)
#   feature_update   : STrack.update_features (per-part EMA)
#   dist_stage1      : appearance distance + fusion, 1st association
#   dist_unconfirmed : appearance distance + fusion, unconfirmed tracks
# ============================================================================

class StageTimer:
    def __init__(self, device=None):
        self.sync = (
            device is not None
            and str(device).lower().startswith("cuda")
            and torch.cuda.is_available()
        )
        self.ms = {}
        self.counts = {}

    def reset(self):
        self.ms = {}
        self.counts = {}

    def _sync(self):
        if self.sync:
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def count(self, name, n=1):
        self.counts[name] = self.counts.get(name, 0) + int(n)


P = PartAppearanceExtractor.NUM_PARTS
D = PartAppearanceExtractor.DIM

# Các tham số của PartAppearanceExtractor: BẮT BUỘC khai báo trong args khi gọi BoTSORT.
# Tên attribute trong args trùng tên tham số của extractor (riêng debug -> args.debug_reid).
EXTRACTOR_PARAMS = (
    "min_bbox_h",
    "pad_ratio",
    "min_keypoint_conf",
    "half_w_ratio",
    "limb_w_ratio",
    "min_limb_ratio",
    "torso_shrink",
    "min_part_area_ratio",
    "min_pixels",
    "good_pixels",
    "l_weight",
    "occ_power",
)


def build_extractor_kwargs(args):
    """Lấy toàn bộ config của PartAppearanceExtractor từ args, không có default."""
    names = list(EXTRACTOR_PARAMS) + ["debug_reid"]
    missing = [n for n in names if not hasattr(args, n)]
    if missing:
        raise ValueError(
            "with_reid=True nhưng args thiếu các tham số của PartAppearanceExtractor: "
            + ", ".join(missing))
    kwargs = {n: getattr(args, n) for n in EXTRACTOR_PARAMS}
    kwargs["debug"] = args.debug_reid
    return kwargs


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    latency_timer = None          # set by BoTSORT.__init__
    feat_alpha = 0.9              # EMA, set by BoTSORT.__init__
    feat_q_min = 0.3              # quality tối thiểu để update EMA

    def __init__(self, tlwh, score, feat=None, keypoints=None):
        """feat = (feat (P,D), valid (P,), qual (P,)) hoặc None"""
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.keypoints = keypoints

        # detection-level (frame hiện tại)
        self.curr_feat = None
        self.curr_valid = None
        self.curr_qual = None
        # track-level (EMA theo part)
        self.smooth_feat = None
        self.smooth_valid = None

        if feat is not None:
            self.update_features(*feat)

    # ------------------------------------------------------------------
    def update_features(self, feat, valid, qual):
        timer = STrack.latency_timer
        if timer is None:
            self._update_features_impl(feat, valid, qual)
        else:
            with timer.stage("feature_update"):
                self._update_features_impl(feat, valid, qual)

    def _update_features_impl(self, feat, valid, qual):
        self.curr_feat, self.curr_valid, self.curr_qual = feat, valid, qual
        if self.smooth_feat is None:
            self.smooth_feat = np.zeros((P, D), np.float32)
            self.smooth_valid = np.zeros(P, bool)
        a0, qmin = STrack.feat_alpha, STrack.feat_q_min
        for p in range(P):
            if not valid[p] or qual[p] < qmin:      # bị che / kém -> không update
                continue
            if not self.smooth_valid[p]:
                self.smooth_feat[p] = feat[p]
                self.smooth_valid[p] = True
                continue
            a = 1.0 - (1.0 - a0) * qual[p]          # quality cao -> update nhanh hơn
            v = a * self.smooth_feat[p] + (1.0 - a) * feat[p]
            self.smooth_feat[p] = v / (np.linalg.norm(v) + 1e-9)

    # ------------------------------------------------------------------
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def _absorb(self, new_track):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.curr_valid, new_track.curr_qual)
        if new_track.keypoints is not None:
            self.keypoints = new_track.keypoints

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        self._absorb(new_track)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """Keypoints lấy từ new_track.keypoints."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        self._absorb(new_track)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    # ------------------------------------------------------------------
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    """
    args (ngoài các args gốc), tất cả đều optional trừ nhóm extractor:
        with_reid          : bật part appearance
        fusion_mode        : 'min' (gốc BoT-SORT) | 'weighted'      (default 'min')
        fusion_lambda      : trọng số IoU khi weighted              (default 0.8)
        appearance_thresh  : ngưỡng gate (trên thang [0,1] đã calib, KHÔNG chia 2)
        part_calib         : (d_same, d_diff) từ eval_discriminability, hoặc None
        feat_alpha         : EMA                                    (default 0.9)
        feat_q_min         : quality tối thiểu để update EMA        (default 0.3)
        match_q_min        : quality tối thiểu của detection để dùng khi match (0.3)
        min_shared         : tỉ lệ trọng số part chung tối thiểu    (default 0.3)
        kps_normalized     : True/False/None(auto) cho toạ độ keypoint

    BẮT BUỘC khi with_reid=True (không có default, truyền vào PartAppearanceExtractor):
        min_bbox_h, pad_ratio, min_keypoint_conf, half_w_ratio, limb_w_ratio,
        min_limb_ratio, torso_shrink, min_part_area_ratio, min_pixels,
        good_pixels, l_weight, occ_power,
        debug_reid         : True -> extractor lưu polygon vào .dbg (map sang debug=)
    """

    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.latency = StageTimer(getattr(args, "device", None))
        STrack.latency_timer = self.latency
        STrack.feat_alpha = getattr(args, "feat_alpha", 0.9)
        STrack.feat_q_min = getattr(args, "feat_q_min", 0.3)

        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.fusion_mode = getattr(args, "fusion_mode", "min")
        self.fusion_lambda = getattr(args, "fusion_lambda", 0.8)
        self.part_calib = getattr(args, "part_calib", None)
        self.match_q_min = getattr(args, "match_q_min", 0.3)
        self.min_shared = getattr(args, "min_shared", 0.3)
        self.kps_normalized = getattr(args, "kps_normalized", None)

        self.extractor = None
        self._warned_no_kps = False
        if args.with_reid:
            self.extractor = PartAppearanceExtractor(**build_extractor_kwargs(args))

        self.last_feats = None   # (feats, qual, valid, occ) của frame gần nhất, để debug

        self.gmc = GMC(method=args.cmc_method)

    def get_frame_latency(self):
        return {"ms": dict(self.latency.ms), "counts": dict(self.latency.counts)}

    # ------------------------------------------------------------------
    @staticmethod
    def _stack_track_feats(tracks):
        Ft = np.zeros((len(tracks), P, D), np.float32)
        Vt = np.zeros((len(tracks), P), bool)
        for i, t in enumerate(tracks):
            if t.smooth_feat is not None:
                Ft[i] = t.smooth_feat
                Vt[i] = t.smooth_valid
        return Ft, Vt

    def _stack_det_feats(self, dets):
        Fd = np.zeros((len(dets), P, D), np.float32)
        Vd = np.zeros((len(dets), P), bool)
        for i, d in enumerate(dets):
            if d.curr_feat is not None:
                Fd[i] = d.curr_feat
                Vd[i] = d.curr_valid & (d.curr_qual >= self.match_q_min)
        return Fd, Vd

    def _fuse(self, tracks, dets, ious_dists, ious_mask, stage):
        """Appearance distance + fusion với IoU. Trả (T,N)."""
        if len(tracks) == 0 or len(dets) == 0:
            return ious_dists
        with self.latency.stage(stage):
            Ft, Vt = self._stack_track_feats(tracks)
            Fd, Vd = self._stack_det_feats(dets)
            emb, known = PartAppearanceExtractor.pairwise_distance(
                Ft, Vt, Fd, Vd, min_shared=self.min_shared,
                calib=self.part_calib, return_mask=True)

            self.latency.count("n_pairs", emb.size)
            self.latency.count("n_pairs_unknown", int((~known).sum()))

            if self.fusion_mode == "weighted":
                use = known & ~ious_mask
                lam = self.fusion_lambda
                dists = np.where(use, lam * ious_dists + (1.0 - lam) * emb, ious_dists)
            else:
                emb = emb.copy()
                emb[emb > self.appearance_thresh] = 1.0
                emb[ious_mask] = 1.0
                self.latency.count("n_emb_better_than_iou", int((emb < ious_dists).sum()))
                dists = np.minimum(ious_dists, emb)
        return dists

    # ------------------------------------------------------------------
    def update(self, output_results, img, keypoints=None):
        """
        output_results : (M,5|6+) như cũ
        keypoints      : (M,17,3) hoặc (M,17,2), CÙNG THỨ TỰ với output_results.
                         None -> chạy thuần IoU.
        """
        self.frame_id += 1
        self.latency.reset()
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        kps_keep = None
        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2

            if keypoints is not None:
                assert len(keypoints) == len(output_results), \
                    "keypoints phải cùng số lượng/thứ tự với output_results"
                kps_all = np.asarray(keypoints)

            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            if keypoints is not None:
                kps_all = kps_all[lowest_inds]

            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            if keypoints is not None:
                kps_keep = kps_all[remain_inds]
        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []

        ''' Extract part appearance '''
        feats_keep = None
        kp_px = None
        if self.args.with_reid and len(dets) > 0:
            if kps_keep is None:
                if not self._warned_no_kps:
                    print("[warn] with_reid=True nhưng không có keypoints -> chạy IoU-only")
                    self._warned_no_kps = True
            else:
                with self.latency.stage("extract"):
                    kp_px = PartAppearanceExtractor.to_pixel_keypoints(
                        kps_keep, img.shape, normalized=self.kps_normalized)
                    f, q, v, occ = self.extractor.extract(
                        img, kp_px, dets, ctx_boxes=bboxes)   # occ tính trên MỌI box
                    feats_keep = (f, q, v)
                    self.last_feats = (f, q, v, occ)
                self.latency.count("n_extracted", len(dets))
                for p in range(P):
                    self.latency.count(f"valid_part{p}", int(v[:, p].sum()))

        if len(dets) > 0:
            if feats_keep is not None:
                f, q, v = feats_keep
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s, (f[i], v[i], q[i]), kp_px[i])
                    for i, (tlbr, s) in enumerate(zip(dets, scores_keep))]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s)
                              for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if feats_keep is not None:
            dists = self._fuse(strack_pool, detections, ious_dists, ious_dists_mask, "dist_stage1")
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes (IoU only)'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
        else:
            dets_second = []
            scores_second = []

        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if feats_keep is not None:
            dists = self._fuse(unconfirmed, detections, ious_dists, ious_dists_mask, "dist_unconfirmed")
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb