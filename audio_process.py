import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, decimate
from collections import deque
from dataclasses import dataclass
import librosa
from read_crossing_head import crossing_list, head_count_df

# ----------------------------
# 0) Config
# ----------------------------
# --- NEW: optional assumed occlusion duration if you only have crossing times (no durations)
ASSUMED_OCC_DUR_SEC = 0.8  # tune: 0.4–0.8 s typical for a single passer-by
AUDIO_PATH = "room_afternoon.wav"   # <-- replace
CROSSING_TIMES_SEC = []            # optional: e.g., [12.3, 12.8, 30.1, ...]
FS_TARGET = 16000                  # 16 kHz (or 8000)
FRAME_MS = 20                      # 20 ms frames
FRAME_HOP_MS = 10                  # 50% overlap => 10 ms hop
WIN_SEC = 10.0                     # 10 s decision window
WIN_HOP_SEC = 5.0                  # 5 s hop (rolling)
DOOR_MASK_SEC = 2.0                # mask ±1.0 s around crossings
RAW_FS = 48000

# Speechiness weights
W_VAF, W_BR, W_SF, W_MOD = 0.35, 0.25, 0.20, 0.20

# Tier thresholds (with small hysteresis if you later add it)
T_LOW_MED  = 0.45
T_MED_HIGH = 0.75

# Crossing Variables & Head Counts as Labels
crossing_list = np.array(crossing_list)
CROSSING_TIMES_SEC = crossing_list[crossing_list < 6960]

@dataclass
class CrossingsPerMinute:
    horizon_s: float = 120.0
    def __post_init__(self):
        self._ev = deque()  # store event start times (floats, seconds)
    def update(self, event_intervals, now: float):
        # Push new event starts seen in this 10 s window
        for (a, _b) in event_intervals:
            self._ev.append(float(a))
        cutoff = now - self.horizon_s
        while self._ev and self._ev[0] < cutoff:
            # print(self._ev)
            self._ev.popleft()
    def crossings_per_min(self) -> float:
        # print(len(self._ev))
        return float(len(self._ev))  # count over last 60 s

@dataclass
class TemporalTracker:
    span_windows: int = 3  # ~30 s with 10 s windows
    def __post_init__(self):
        self.alpha = 2.0 / (self.span_windows + 1.0)  # EMA coeff (span=3 → 0.5)
        self.prev_vaf = None
        self.prev_cross_rate = None
        self.ewma_vaf = None
        self.ewma_rms = None
    def update(self, vaf_t: float, rms_db_t: float, cross_rate_t: float):
        d_vaf = 0.0 if self.prev_vaf is None else (vaf_t - self.prev_vaf)
        d_cross = 0.0 if self.prev_cross_rate is None else (cross_rate_t - self.prev_cross_rate)
        self.ewma_vaf = vaf_t if self.ewma_vaf is None else \
            (self.alpha * vaf_t + (1.0 - self.alpha) * self.ewma_vaf)
        self.ewma_rms = rms_db_t if self.ewma_rms is None else \
            (self.alpha * rms_db_t + (1.0 - self.alpha) * self.ewma_rms)
        self.prev_vaf = vaf_t
        self.prev_cross_rate = cross_rate_t
        return {'ΔVAF': float(d_vaf), 'ΔCrossings': float(d_cross),
                'EWMA_VAF': float(self.ewma_vaf), 'EWMA_RMSdB': float(self.ewma_rms)}

def get_features(path):
# ----------------------------
# 1) Load + mono + resample (if needed)
# ----------------------------

    fs, data = wavfile.read(path)          # int16 or float
    data = data.astype(np.float32)[: 6960*RAW_FS]
    if data.ndim == 2:                           # downmidata stereo
        data = data.mean(axis=1)

    # normalize to [-1,1] if int-like
    if data.dtype != np.float32:
        maxv = np.max(np.abs(data))
        quantile = np.percentile(np.abs(data), 95)
        if maxv > 0: data = data / maxv
        # data = data / quantile
    # Simple resample via decimation/interpolation (use exact resampler if you like)

    # x, fs = simple_resample(x, fs, FS_TARGET)
    data = librosa.resample(data, orig_sr=fs, target_sr=FS_TARGET)
    fs = FS_TARGET

    # ----------------------------
    # 2) Band-pass 300–3000 Hz + high-pass protection
    # ----------------------------
    def butter_bandpass(low, high, target_fs, order=4):
        nyq = target_fs * 0.5
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return b, a

    b_bp, a_bp = butter_bandpass(300.0, 3000.0, fs, order=4)
    data_bp = filtfilt(b_bp, a_bp, data)  # zero-phase to avoid phase distortion

    # ----------------------------
    # 3) Frame the signal (20 ms, 50% overlap)
    # ----------------------------
    frame_len = int(round(FRAME_MS * 1e-3 * fs))     # samples per frame
    hop_len   = int(round(FRAME_HOP_MS * 1e-3 * fs)) # samples per hop
    n_frames  = 1 + max(0, (len(data_bp) - frame_len) // hop_len)

    # Hann window for FFT features
    hann = 0.5 - 0.5*np.cos(2*np.pi*np.arange(frame_len)/frame_len)

    # Frame centers in seconds (for door masking)
    frame_centers_sec = (np.arange(n_frames)*hop_len + frame_len/2) / fs

    # Door mask: frames near crossings are ignored
    mask_frames = np.ones(n_frames, dtype=bool)
    if len(CROSSING_TIMES_SEC) > 0:
        crossing = np.array(CROSSING_TIMES_SEC)[None, :]
        frame_sec = frame_centers_sec[:, None]
        near = np.any(np.abs(frame_sec - crossing) <= DOOR_MASK_SEC, axis=1)
        mask_frames = ~near  # True = keep, False = mask out

    # ----------------------------
    # 4) Per-frame features
    # ----------------------------
    def spectral_bins(freq_lo, freq_hi, sample_rate, n_fft):
        freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
        idx = np.where((freqs >= freq_lo) & (freqs <= freq_hi))[0]
        return idx

    idx_voice = spectral_bins(300, 3000, fs, frame_len)
    idx_low   = spectral_bins(50, 150, fs, frame_len)

    # Per-frame ZCR (on band-passed frame s_win)
    def zcr(sig):
        # sig is the windowed frame (e.g., s_win)
        s_ = np.signbit(sig)
        return np.mean(s_[1:] != s_[:-1])

    # Two-band spectral tilt in dB (high vs mid speech bands)
    # Using the same PSD you already computed per frame.
    # idx_mid and idx_high are frequency-bin index arrays you define once.
    def two_band_tilt_db(PSD_, idx_mid_, idx_high_, eps=1e-12):
        e_mid  = PSD_[idx_mid_].sum() + eps    # ~300–1000 Hz
        e_high = PSD_[idx_high_].sum() + eps    # ~1000–3000 Hz
        return 10.0 * np.log10(e_high / e_mid)

    def rms_db_window(sig: np.ndarray, ref: float = 1.0,
                      clip_min_db: float = -60.0, clip_max_db: float = 0.0) -> float:
        eps = 1e-12
        rms = float(np.sqrt(np.mean(sig.astype(np.float32)**2) + eps))
        db = 20.0 * np.log10(max(rms, eps) / ref)
        return float(np.clip(db, clip_min_db, clip_max_db))


    # Pre-compute PSD-bin indices once (after you know fs and frame_len)
    idx_mid   = spectral_bins(300, 1000, fs, frame_len)   # ~mid speech band
    idx_high  = spectral_bins(1000, 3000, fs, frame_len)  # ~high speech band

    # Define per-frame metrics
    ZCR_f      = np.zeros(n_frames, dtype=np.float32)
    Tilt2B_dbf = np.zeros(n_frames, dtype=np.float32)
    VAF_frames = np.zeros(n_frames, dtype=bool)
    BandRatio_f = np.zeros(n_frames, dtype=np.float32)
    SpecFlat_f  = np.zeros(n_frames, dtype=np.float32)

    # Hilbert envelope for Mod3_8Hz will be computed per 10 s window (more stable),
    # but we also keep per-frame RMS to build a robust floor.
    RMS_f = np.zeros(n_frames, dtype=np.float32)

    eps = 1e-12
    for i in range(n_frames):
        s = data_bp[i*hop_len : i*hop_len + frame_len]
        s_raw = data[i*hop_len : i*hop_len + frame_len]
        if len(s) < frame_len:
            break
        s_win = s * hann
        s_win_raw = s_raw * hann
        # RMS
        rms = np.sqrt(np.mean(s_win**2) + eps)
        RMS_f[i] = rms
        # FFT
        S = np.fft.rfft(s_win, n=frame_len)
        S_raw = np.fft.rfft(s_win_raw, n=frame_len)
        PSD = (np.abs(S)**2) / frame_len
        PSD_raw = (np.abs(S_raw)**2) / frame_len

        # ZCR on the (band-passed, windowed) time frame
        ZCR_f[i] = zcr(s_win)
        # Two-band spectral tilt (dB)
        Tilt2B_dbf[i] = two_band_tilt_db(PSD, idx_mid, idx_high)

        # BandRatio
        e_voice = PSD_raw[idx_voice].sum() + eps
        e_low   = PSD_raw[idx_low].sum()   + eps
        BandRatio_f[i] = e_voice / e_low

        # Spectral flatness: gm/mean over the *voice band* (robust to LF thuds)
        voice_band = PSD_raw[idx_voice] + eps
        gm = np.exp(np.mean(np.log(voice_band)))
        am = np.mean(voice_band)
        SpecFlat_f[i] = float(np.clip(gm / am, 0.0, 1.0))

    # Voice-Activity Fraction (VAF) — need a robust energy floor + persistence
    # robust floor from masked frames only

    # valid_rms = RMS_f[mask_frames]
    # # median + 1.5*MAD
    # floor = np.median(valid_rms) + 1.5*np.median(np.abs(valid_rms - np.median(valid_rms)))
    #
    # # simple "speechy" test: energy above floor and BandRatio above a minimal cutoff
    # speechy = (RMS_f > floor) & (BandRatio_f > 2.0)

    # --- Energy floor (robust) and adaptive bounds ---

    # Use only unmasked frames to estimate floors (mask_frames: True = keep)
    valid = mask_frames.copy()
    valid &= np.isfinite(RMS_f) & np.isfinite(ZCR_f) & np.isfinite(Tilt2B_dbf)

    # Robust energy floor = median + 1.5*MAD
    rms_valid = RMS_f[valid]
    rms_floor = np.median(rms_valid) + 1.5*np.median(np.abs(rms_valid - np.median(rms_valid)))

    # Adaptive ZCR band around room behavior (clipped to hard limits)
    zcr_valid = ZCR_f[valid]
    zcr_med   = np.median(zcr_valid)
    zcr_q1, zcr_q3 = np.percentile(zcr_valid, [25, 75])
    zcr_iqr   = max(zcr_q3 - zcr_q1, 1e-6)

    ZCR_MIN_HARD, ZCR_MAX_HARD = 0.03, 0.30      # hard safety bounds
    ZCR_LOW  = max(ZCR_MIN_HARD, zcr_med - 1.0*zcr_iqr)
    ZCR_HIGH = min(ZCR_MAX_HARD, zcr_med + 1.5*zcr_iqr)

    # Spectral tilt threshold (two-band dB). Start with -3 dB; you can adapt via percentiles if needed.
    TILT2B_MIN_DB = -3.0

    # --- Frame-wise "speechy" test: energy AND ZCR AND spectral tilt ---

    energy_gate = (RMS_f > rms_floor)
    zcr_gate    = (ZCR_f >= ZCR_LOW) & (ZCR_f <= ZCR_HIGH)
    tilt_gate   = (Tilt2B_dbf >= TILT2B_MIN_DB)

    speechy_raw = energy_gate & zcr_gate & tilt_gate

    # Never count masked frames (near ultrasonic crossings)
    speechy_raw = speechy_raw & mask_frames

    # persistence: require >=3 consecutive frames to be speechy
    def apply_persistence(bits, run=3):
        y = bits.copy()
        count = 0
        for i in range(len(bits)):
            count = count + 1 if bits[i] else 0
            if bits[i] and count < run:
                y[i] = False
        # also back-fill the previous run-1 frames in each run
        # (optional; conservative choice: leave as-is)
        return y

    VAF_frames = apply_persistence(speechy_raw, run=3)
    VAF_frames = VAF_frames & mask_frames  # never count masked frames

    # ----------------------------
    # 5) Decision windows (10 s, hop 5 s) and window-level features
    # ----------------------------
    win_len_frames = int(round(WIN_SEC / (FRAME_HOP_MS*1e-3) / 2)) * 2  # approx frames in 10 s
    win_hop_frames = int(round(WIN_HOP_SEC / (FRAME_HOP_MS*1e-3)))

    # Helper: robust normalization 0..1 (per session)

    def robust_norm(x):
        median = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        if iqr < 1e-9:
            iqr = 1.0
        z = (x - median) / (iqr)
        # squash to 0..1 via logistic-like mapping; here clip a linear map
        z = 0.5 + 0.2*z  # spread
        return np.clip(z, 0, 1)

    # Build arrays for window outputs
    # -------------------------
    # Robust scaler helpers
    # -------------------------
    def fit_robust_scaler(x: np.ndarray):
        """Return (median, IQR) for robust scaling; guard IQR to avoid divide-by-zero."""
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        if iqr < 1e-6:
            iqr = 1.0
        return float(med), iqr

    def transform_robust01(x: np.ndarray, med: float, iqr: float, clip: float = 3.0):
        """Robust z → clip to [-clip, clip] → map to [0,1]."""
        z = (np.asarray(x, dtype=float) - med) / iqr
        z = np.clip(z, -clip, clip)
        return (z + clip) / (2.0 * clip)

    # -------------------------
    # Choose which features to scale vs keep
    # -------------------------
    # NEW features to scale (unbounded or wide-range)


    win_centers_sec, VAF_win, BR_win, SF_win, MOD_win, SS_win = [], [], [], [], [], []
    RMSdB_win = []
    CrossCount_win = []
    DwellRatio_win = []
    CrossPerMin_win = []
    DeltaVAF_win = []
    DeltaCross_win = []
    EWMA_VAF_win = []
    EWMA_RMSdB_win = []

    # --- Trackers
    xpm = CrossingsPerMinute(horizon_s=60.0)
    tt  = TemporalTracker(span_windows=3)

    # Convenience: numpy arrays for crossings & (optional) intervals
    _cross_times = np.array(CROSSING_TIMES_SEC, dtype=float) if len(CROSSING_TIMES_SEC) else np.array([], dtype=float)
    # _intervals = np.array(crossing_intervals, dtype=float) if len(crossing_intervals) else np.empty((0,2), dtype=float)
    _intervals = np.empty((0,2), dtype=float)

    # Precompute analytic envelope for Mod3_8Hz at full rate
    analytic = hilbert(data_bp)
    envelope = np.abs(analytic)
    # Decimate envelope to ~100 Hz for modulation analysis
    decim_factor = max(1, int(fs // 100))
    env_100 = decimate(envelope, decim_factor, zero_phase=True)
    fs_env = fs / decim_factor

    def bandpower_env_3_8Hz(env_seg, env_fs):
        # FFT the envelope segment and integrate 3–8 Hz band
        n = len(env_seg)
        if n < 10: return 0.0
        E = np.fft.rfft(env_seg * np.hanning(n))
        P = (np.abs(E)**2) / n
        freqs = np.fft.rfftfreq(n, 1 / env_fs)
        band = (freqs >= 3.0) & (freqs <= 8.0)
        total_P = P.sum() + 1e-12
        return float(P[band].sum() / total_P)

    # slide windows
    i = 0
    while i + win_len_frames <= n_frames:
        f_start = i
        f_end = i + win_len_frames
        # indices of frames in this 10 s window
        idx = np.arange(f_start, f_end)
        # Aggregate frame features (masked frames already removed from VAF calc)
        VAF_w = np.mean(VAF_frames[idx])
        band_ratio_w  = np.median(BandRatio_f[idx])   # robust
        SpecFlat_w  = np.median(SpecFlat_f[idx])

        # Mod3_8Hz from the envelope segment covering this 10 s span
        t_start = (f_start * hop_len) / fs
        t_end   = (f_end * hop_len + frame_len) / fs
        e0 = int(round(t_start * fs_env))
        e1 = int(round(t_end   * fs_env))
        mod = bandpower_env_3_8Hz(env_100[max(0, e0):min(len(env_100), e1)], fs_env)

        # --- NEW: RMS_dB for this 10 s window (use band-passed signal to emphasize speech)
        s0 = int(round(t_start * fs))
        s1 = int(round(t_end * fs))
        seg = data_bp[max(0, s0):min(len(data_bp), s1)]
        rmsdb = rms_db_window(seg)
        RMSdB_win.append(rmsdb)

        # --- NEW: CrossCount & DwellRatio for this window
        # CrossCount: count crossing times that fall inside [t_start, t_end)
        if _cross_times.size:
            cross_count = int(((_cross_times >= t_start) & (_cross_times < t_end)).sum())
        else:
            cross_count = 0

        # DwellRatio: if we have intervals, use true overlap;
        # otherwise approximate with assumed duration per crossing.
        # if _intervals.size:
        #     # sum of overlaps between [t_start,t_end) and each (a,b)
        #     overlap = 0.0
        #     for a, b in _intervals:
        #         if b <= t_start or a >= t_end:
        #             continue
        #         overlap += (min(b, t_end) - max(a, t_start))
        #     dwell = float(np.clip(overlap / max(1e-9, (t_end - t_start)), 0.0, 1.0))
        # else:
        dwell = float(np.clip((cross_count * ASSUMED_OCC_DUR_SEC) / max(1e-9, (t_end - t_start)), 0.0, 1.0))

        CrossCount_win.append(cross_count)
        DwellRatio_win.append(dwell)

        # --- NEW: Crossings/min (rolling 60 s) via tracker
        # Build "events" for this window; if only times are known, use (t, t)
        if _intervals.size:
            events_now = [(a, b) for a, b in _intervals if (a >= t_start and a < t_end)]
        else:
            events_now = [(float(t), float(t)) for t in _cross_times if (t >= t_start and t < t_end)]

        # print(events_now)
        xpm.update(events_now, now=t_end)
        cpm = xpm.crossings_per_min()
        # print(cpm)
        CrossPerMin_win.append(cpm)

        # --- NEW: Temporal augments (ΔVAF, ΔCrossings, EWMAs)
        temporal_out = tt.update(vaf_t=VAF_w, rms_db_t=rmsdb, cross_rate_t=cpm)
        DeltaVAF_win.append(temporal_out['ΔVAF'])
        DeltaCross_win.append(temporal_out['ΔCrossings'])
        EWMA_VAF_win.append(temporal_out['EWMA_VAF'])
        EWMA_RMSdB_win.append(temporal_out['EWMA_RMSdB'])
        # Normalize each feature 0..1 per session
        # (Do this after collecting all windows; here we collect raw first.)
        VAF_win.append(VAF_w)
        BR_win.append(band_ratio_w)
        SF_win.append(SpecFlat_w)
        MOD_win.append(mod)
        win_centers_sec.append((t_start + t_end)/2)
        i += win_hop_frames

    VAF_win = np.array(VAF_win); BR_win = np.array(BR_win)
    SF_win  = np.array(SF_win);  MOD_win = np.array(MOD_win)
    RMSdB_win       = np.array(RMSdB_win, dtype=np.float32)
    CrossCount_win  = np.array(CrossCount_win, dtype=np.int32)
    DwellRatio_win  = np.array(DwellRatio_win, dtype=np.float32)
    CrossPerMin_win = np.array(CrossPerMin_win, dtype=np.float32)
    DeltaVAF_win    = np.array(DeltaVAF_win, dtype=np.float32)
    DeltaCross_win  = np.array(DeltaCross_win, dtype=np.float32)
    EWMA_VAF_win    = np.array(EWMA_VAF_win, dtype=np.float32)
    EWMA_RMSdB_win  = np.array(EWMA_RMSdB_win, dtype=np.float32)


    to_scale = {
        "RMS_dB":         RMSdB_win,         # e.g., [-60, 0] dBFS after clipping
        # "CrossPerMin":   CrossPerMin_win,   # events/min over rolling 60 s
        "DeltaVAF":      DeltaVAF_win,      # roughly [-1, 1]
        "DeltaCross":    DeltaCross_win,    # small, centered around 0
        "EWMA_RMSdB":    EWMA_RMSdB_win,    # smoothed dB
    }

    # Already bounded in [0,1] — keep, just clip (optional)
    bounded_keep = {
        "VAF":           VAF_win,           # if you have it
        "DwellRatio":    DwellRatio_win,    # fraction of time occluded
        "EWMA_VAF":      EWMA_VAF_win,      # EMA of VAF
        "Mod3_8Hz":      MOD_win,           # if computed as normalized modulation energy
        "SpecFlat":      SF_win
    }

    # --- New robust scaling to [0,1] for (VAF, BR_dB, SpecFlatness, Mod3_8Hz) ---
    # Choose what data to use to "fit" the scaler (e.g., all windows or a calibration slice)
    cal_mask = slice(None)  # or use a boolean mask on win_centers_sec for first N minutes

    eps = 1e-12

    # 1) BandRatio → dB (preferred to log1p) then robust 0..1
    BR_win = np.asarray(BR_win, float)
    BR_dB  = 10.0 * np.log10(BR_win + eps)        # compress heavy tails
    med_br, iqr_br = fit_robust_scaler(BR_dB[cal_mask])
    BR_01  = transform_robust01(BR_dB, med_br, iqr_br, clip=3.0)

    # 2) Robust-scale Crossings/min → [0,1]
    med_cpm, iqr_cpm = fit_robust_scaler(np.asarray(CrossPerMin_win, float)[cal_mask])
    CrossPerMin_01   = transform_robust01(CrossPerMin_win, med_cpm, iqr_cpm, clip=3.0)

    # 2) Make CrossCount comparable: map this window’s count to a per-minute equivalent
    #    (10 s window → multiply by 6), then reuse the SAME scaler as Crossings/min.
    CrossCount_per_min_equiv = np.asarray(CrossCount_win, float) * 6.0
    CrossCount_01 = transform_robust01(CrossCount_per_min_equiv, med_cpm, iqr_cpm, clip=3.0)

    # -------------------------
    # Pick a calibration slice (e.g., first few minutes or training set)
    # -------------------------
    # If you have window center times: use a mask like:
    # cal_mask = (win_centers_sec >= t0) & (win_centers_sec < t1)
    # Otherwise, just use all windows to fit:

    # -------------------------
    # Fit scalers & transform
    # -------------------------
    scalers = {}
    scaled = {}

    for name, arr in to_scale.items():
        arr = np.asarray(arr, dtype=float)
        med, iqr = fit_robust_scaler(arr[cal_mask])
        scalers[name] = {"median": med, "iqr": iqr}
        scaled[name] = transform_robust01(arr, med, iqr, clip=3.0)

    # For bounded features, just clip to [0,1] (keeps semantics)
    for name, arr in bounded_keep.items():
        arr = np.asarray(arr, dtype=float)
        scaled[name] = np.clip(arr, 0.0, 1.0)

    scalers['BandRatio'] = BR_01; scalers['CrossPerMin'] = CrossPerMin_win
    scaled['CrossCount'] = CrossCount_01

    return scalers, scaled, win_centers_sec


# Speechiness Score (0..1)
# SS_win = W_VAF*scaled['VAF'] + W_BR*BR_01 + W_SF*scaled['SpecFlat'] + W_MOD*scaled["Mod3_8Hz"]

# ----------------------------
# 6) Optional: Flow from ultrasonic crossings
# ----------------------------
# Flow_per_min = np.zeros_like(SS_win)
# if len(CROSSING_TIMES_SEC) > 0:
#     crossing = np.array(CROSSING_TIMES_SEC)
#     # For each 10 s window, count crossings in the last 60 s
#     for k, c in enumerate(win_centers_sec):
#         Flow_per_min[k] = ((crossing >= (c - 60)) & (crossing <= c)).sum()

# ----------------------------
# 7) Map Speechiness to tiers (Low/Med/High) with simple hysteresis
# ----------------------------
# tiers = []
# state = 0
# for s in SS_win:
#     if state == "LOW":
#         if s >= T_LOW_MED: state = "MED"
#     elif state == "MED":
#         if s >= T_MED_HIGH: state = "HIGH"
#         elif s < T_LOW_MED - 0.10: state = "LOW"   # 0.10 hysteresis band
#     elif state == "HIGH":
#         if s < T_MED_HIGH - 0.15: state = "MED"    # larger band for stability
#     tiers.append(state)

# At this point you have:
# - win_centers_sec : timestamps for each decision
# - SS_win          : Speechiness score
# - Flow_per_min    : optional flow context
# - tiers           : final Low/Med/High per window

if __name__ == "__main__":
    scalers, bound_keep, win_sec = get_features('room_afternoon.wav')
