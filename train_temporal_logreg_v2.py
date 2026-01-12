# train_temporal_logreg.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from itertools import product
from collections import deque
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import joblib
import json

from read_crossing_head import head_count_df
from audio_process import get_features

# =========================
# 0) CONFIG
# =========================
WIN_SEC = 10.0           # your analysis window length
HOP_SEC = 5.0            # if you hop every 5 s, adjust if different
LABEL_TIER_NAMES = ["Low", "Medium", "High"]
TIER2ID = {"Low": 0, "Medium": 1, "High": 2}

# =========================
# 1) BRING YOUR FEATURES
# =========================
def load_feature_arrays() -> Dict[str, np.ndarray]:
    """
    Replace this with how you load/import arrays from your feature script.
    All arrays must have the same length N (number of windows).
    Required keys shown; include others if you have them.
    """
    # ---------- YOUR_DATA_HERE ----------
    # Example placeholders — REPLACE with your real arrays:
    # win_centers_sec = np.load("win_centers_sec.npy")
    # VAF_win         = np.load("VAF_win.npy")
    # BR_win          = np.load("BandRatio_win.npy")
    # SF_win          = np.load("SpecFlatness_win.npy")
    # MOD_win         = np.load("Mod3_8Hz_win.npy")
    # RMSdB_win       = np.load("RMSdB_win.npy")
    # CrossPerMin_win = np.load("CrossPerMin_win.npy")
    # CrossCount_win  = np.load("CrossCount_win.npy")
    # DwellRatio_win  = np.load("DwellRatio_win.npy")
    # DeltaVAF_win    = np.load("DeltaVAF_win.npy")
    # DeltaCross_win  = np.load("DeltaCross_win.npy")
    # EWMA_VAF_win    = np.load("EWMA_VAF_win.npy")
    # EWMA_RMSdB_win  = np.load("EWMA_RMSdB_win.npy")
    with_scalers, bound_keep, win_sec = get_features('room_afternoon.wav')
    features = with_scalers | bound_keep
    features['win_centers_sec'] = np.array(win_sec).astype(float)
    # with_scalers = {
    #     "RMSdB", "CrossPerMin", "DeltaVAF", "DeltaCross", "EWMA_RMSdB", 'BandRatio',
    # }
    # bounded_keep = {
    #     "VAF", "DwellRatio", "EWMA_VAF", "Mod3_8Hz", "SpecFlat",'CrossCount'
    # }



    # raise NotImplementedError("Wire your arrays here and return the dict below")
    # print(features.head(10))
    return features
    # return {
    #   "win_centers_sec": win_centers_sec.astype(float),
    #   "VAF": VAF_win.astype(float),
    #   "BandRatio": BR_win.astype(float),
    #   "SpecFlatness": SF_win.astype(float),
    #   "Mod3_8Hz": MOD_win.astype(float),
    #   "RMS_dB": RMSdB_win.astype(float),
    #   "CrossPerMin": CrossPerMin_win.astype(float),
    #   "CrossCount": CrossCount_win.astype(float),
    #   "DwellRatio": DwellRatio_win.astype(float),
    #   "DeltaVAF": DeltaVAF_win.astype(float),
    #   "DeltaCross": DeltaCross_win.astype(float),
    #   "EWMA_VAF": EWMA_VAF_win.astype(float),
    #   "EWMA_RMSdB": EWMA_RMSdB_win.astype(float),
    # }


# =========================
# 2) ALIGN PER-SECOND LABELS TO WINDOWS
# =========================
def align_labels_to_windows(
    win_centers_sec: np.ndarray,
    label_times_sec: np.ndarray,
    label_tiers_per_sec: np.ndarray,
    win_sec: float = WIN_SEC
) -> np.ndarray:
    """
    Majority-vote the 1 Hz labels inside each window [center - win_sec/2, center + win_sec/2).
    label_tiers_per_sec should be already in {Low, Medium, High} (strings) or mapped ints.
    Returns y_win as integer class ids (0/1/2).
    """
    half = win_sec / 2.0
    # If labels are strings, map to ints once:
    if label_tiers_per_sec.dtype.kind in {"U", "S", "O"}:
        label_ids = np.array([TIER2ID.get(s, -1) for s in label_tiers_per_sec])
    else:
        label_ids = label_tiers_per_sec.astype(int)

    y = np.full(len(win_centers_sec), fill_value=-1, dtype=int)
    for i, c in enumerate(win_centers_sec):
        a, b = c - half, c + half
        m = (label_times_sec >= a) & (label_times_sec < b)
        if not np.any(m):
            continue
        vals, counts = np.unique(label_ids[m], return_counts=True)
        dom = int(vals[np.argmax(counts)])
        y[i] = dom
    # Drop windows with no coverage (-1) later.
    return y

# =========================
# 3) BUILD DATAFRAME
# =========================
def build_dataframe(feat: Dict[str, np.ndarray], y_win: np.ndarray) -> pd.DataFrame:
    """
    Combine features + labels, drop unlabeled windows, and return a tidy DataFrame.
    """
    N = len(feat["win_centers_sec"])
    df = pd.DataFrame({
        "t_center": feat["win_centers_sec"],
        "y": y_win,
        "VAF": feat["VAF"],
        "BandRatio": feat["BandRatio"],
        "SpecFlat": feat["SpecFlat"],
        "Mod3_8Hz": feat["Mod3_8Hz"],
        "RMS_dB": feat["RMS_dB"],
        "CrossPerMin": feat["CrossPerMin"],
        "CrossCount": feat["CrossCount"],
        "DwellRatio": feat["DwellRatio"],
        "DeltaVAF": feat["DeltaVAF"],
        "DeltaCross": feat["DeltaCross"],
        "EWMA_VAF": feat["EWMA_VAF"],
        "EWMA_RMSdB": feat["EWMA_RMSdB"],
    })
    df = df[df["y"] >= 0].reset_index(drop=True)   # drop unlabeled windows
    return df

# =========================
# 4) TRAIN / EVAL SPLIT (TIME-AWARE)
# =========================
def time_holdout_split(df: pd.DataFrame, holdout_frac: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: first (1-holdout_frac) for train, last holdout_frac for validation.
    """
    df = df.sort_values("t_center").reset_index(drop=True)
    cut = int(round(len(df) * (1.0 - holdout_frac)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

# =========================
# 5) TRAIN PIPELINE
# =========================
def train_temporal_logreg(df_train: pd.DataFrame) -> Pipeline:
    # X_cols = [
    #     "VAF","BandRatio","SpecFlat","Mod3_8Hz","RMS_dB",
    #     "CrossPerMin","CrossCount","DwellRatio",
    #     "DeltaVAF","DeltaCross","EWMA_VAF","EWMA_RMSdB"
    # ]
    X_cols = [
        "VAF", "BandRatio", "SpecFlat", "Mod3_8Hz", "RMS_dB",
        "CrossCount", "DwellRatio",
        "DeltaVAF", "DeltaCross", "EWMA_VAF", "EWMA_RMSdB"
    ]
    X = df_train[X_cols].values.astype(np.float32)
    y = df_train["y"].values.astype(int)

    # Class weights handle imbalance between Low/Med/High
    classes = np.array([0,1,2], dtype=int)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    pipe = Pipeline([
        ("scaler", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=0.7,            # regularization strength; tune 0.1–1.0
            max_iter=200,
            class_weight=class_weight,
            n_jobs=None
        ))
    ])
    pipe.fit(X, y)
    return pipe, X_cols

# =========================
# 6) HYSTERESIS (OPTIONAL, BUT RECOMMENDED)
# =========================
# @dataclass
# class Hysteresis:
#     enter_med: float = 0.60
#     exit_low:  float = 0.45
#     enter_high: float = 0.65
#     exit_med:   float = 0.7
#     relax_if_flow: float = 0.30  # relax thresholds when flow high
#
#     def predict(self, proba: np.ndarray, cross_per_min: np.ndarray) -> np.ndarray:
#         """
#         proba: Nx3 array (cols in order [Low, Med, High])
#         cross_per_min: length-N array for flow gating
#         Returns: integer states per window (0/1/2)
#         """
#         N = len(proba)
#         state = np.zeros(N, dtype=int)  # start in Low
#         for i in range(N):
#             p_low, p_med, p_high = proba[i]
#             # Flow-aware adjustment
#             adj = self.relax_if_flow if cross_per_min[i] >= 5.0 else 0.0  # tweak threshold if lots of crossings
#             if state[i-1] if i>0 else 0 in (0,):  # currently Low
#                 if (p_med + p_high) >= (self.enter_med - adj):
#                     state[i] = 1
#                 else:
#                     state[i] = 0
#             elif state[i-1] == 1:  # currently Medium
#                 if p_high >= (self.enter_high - adj - 0.1):
#                     state[i] = 2
#                 elif (p_med + p_high) <= self.exit_low:
#                     state[i] = 0
#                 else:
#                     state[i] = 1
#             else:  # currently High
#                 if p_high <= self.exit_med:
#                     state[i] = 1
#                 else:
#                     state[i] = 2
#         return state

# -----------------------
# 1) EMA smoothing on probabilities (flow-aware)
# -----------------------
def decode_ema(proba: np.ndarray, flow: np.ndarray,
               alpha_lowflow=0.85, alpha_highflow=0.15, flow_thresh=4.0):
    """
    proba: Nx3 softmax outputs for [Low, Med, High]
    flow:  length-N CrossPerMin (ultrasonic)
    alpha_lowflow: heavier smoothing when flow is quiet
    alpha_highflow: lighter smoothing when flow is busy (react faster)
    """
    N, K = proba.shape
    P = np.zeros_like(proba)
    P[0] = proba[0]
    for i in range(1, N):
        alpha = alpha_highflow if flow[i] >= flow_thresh else alpha_lowflow
        P[i] = alpha * P[i-1] + (1.0 - alpha) * proba[i]
    return np.argmax(P, axis=1), P

# -----------------------
# 2) Margin-based switch (no absolute thresholds)
# -----------------------
def decode_margin(proba: np.ndarray, base_margin=0.08,
                  extra_margin_when_quiet=0.02, flow=None, flow_thresh=4.0):
    """
    Switch to a new class only if p_new - p_current >= margin.
    When flow is quiet (few crossings), add a small extra margin to damp flips.
    """
    N, K = proba.shape
    y = np.zeros(N, dtype=int)
    y[0] = int(np.argmax(proba[0]))
    for i in range(1, N):
        cur = y[i-1]
        margin = base_margin
        if flow is not None and flow[i] < flow_thresh:
            margin += extra_margin_when_quiet
        best = int(np.argmax(proba[i]))
        if best != cur and (proba[i, best] - proba[i, cur]) >= margin:
            y[i] = best
        else:
            y[i] = cur
    return y

# -----------------------
# 3) Switching-cost decoder (online Viterbi-lite)
# -----------------------
def decode_switch_cost(proba: np.ndarray, flow: np.ndarray,
                       lambda_base=0.1, lambda_low=0.05, flow_thresh=4.0):
    """
    Maximize cumulative score sum_t [ log p_t(y_t) ] - sum_t [ lambda_t * 1(y_t != y_{t-1}) ]
    with lambda_t reduced when flow is high (easier to flip).
    Greedy online approximation (works well in practice).
    """
    N, K = proba.shape
    y = np.zeros(N, dtype=int)
    y[0] = int(np.argmax(proba[0]))
    for i in range(1, N):
        lam = lambda_low if flow[i] >= flow_thresh else lambda_base
        # Score staying vs switching to each k
        stay_score = np.log(proba[i, y[i-1]] + 1e-9)  # no penalty
        switch_scores = np.log(proba[i] + 1e-9) - lam
        # Choose best among stay and switches
        if stay_score >= switch_scores.max():
            y[i] = y[i-1]
        else:
            y[i] = int(np.argmax(switch_scores))
    return y

# -----------------------
# 4) K-of-M majority vote on classes
# -----------------------
def decode_majority(proba: np.ndarray, k=3, m=5):
    """
    Predict class by majority vote over last M argmaxes (2-of-3 default).
    """
    raw = np.argmax(proba, axis=1)
    N = len(raw)
    y = np.zeros(N, dtype=int)
    buf = deque([], maxlen=m)
    for i in range(N):
        buf.append(int(raw[i]))
        # count occurrences in buffer
        counts = np.bincount(list(buf), minlength=3)
        if counts.max() >= k:
            y[i] = int(np.argmax(counts))
        else:
            y[i] = int(raw[i])
    return y

# -----------------------
# Example usage inside your evaluation
# -----------------------
def decode_without_hysteresis(pipe, X_cols, df, method="ema"):
    X = df[X_cols].values.astype(np.float32)
    P = pipe.predict_proba(X)
    flow = df["CrossPerMin"].values.astype(float)

    if method == "ema":
        y_hat, P_smooth = decode_ema(P, flow)
    elif method == "margin":
        y_hat = decode_margin(P, base_margin=0.08, extra_margin_when_quiet=0.04, flow=flow)
    elif method == "switchcost":
        y_hat = decode_switch_cost(P, flow, lambda_base=0.12, lambda_low=0.05)
    elif method == "majority":
        y_hat = decode_majority(P, k=2, m=3)
    else:
        y_hat = np.argmax(P, axis=1)  # raw argmax
    return y_hat

def one_way_flow(raw_pred):
    process_pred = raw_pred

    buffer = deque([], maxlen=3)
    state = 0
    for i in range(len(process_pred)):
        if i <= 25 or i >= len(process_pred)*0.91:
            continue
        buffer.append(process_pred[i])
        counts = np.bincount(list(buffer), minlength=3)
        potential_state = np.argmax(counts)
        if potential_state > state and counts.max()>2 and state==0:
            state = potential_state
        elif max(buffer) > state and state==1:
            state = max(buffer)

        if process_pred[i] < state:
            raw_pred[i] = state

    return raw_pred

# =========================
# 7) EVALUATION
# =========================
def evaluate(pipe: Pipeline, X_cols: List[str], df: pd.DataFrame, method="margin") -> Dict:
    X = df[X_cols].values.astype(np.float32)
    y_true = df["y"].values.astype(int)
    # proba = pipe.predict_proba(X)
    # y_pred_plain = np.argmax(proba, axis=1)
    # # print(proba[:500])
    # print(y_pred_plain)
    # if use_hysteresis:
    #     hyst = Hysteresis()
    #     y_pred = hyst.predict(proba, df["CrossPerMin"].values.astype(float))
    # else:
    #     y_pred = y_pred_plain
    y_pred = decode_without_hysteresis(pipe, X_cols, df, method=method)
    y_pred = one_way_flow(y_pred)

    output_compare = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, 'time':df['t_center']})
    output_compare.to_csv("compare_y_pred_y_true.csv", index=False)


    report = classification_report(y_true, y_pred, target_names=LABEL_TIER_NAMES, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    f1_macro = f1_score(y_true, y_pred, average="macro")

    return {
        "macro_f1": float(f1_macro),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        # "y_pred_plain": y_pred_plain.tolist(),
    }

# =========================
# 8) SAVE / LOAD PARAMS
# =========================
def save_pipeline(pipe: Pipeline, X_cols: List[str], path_prefix: str):
    joblib.dump({"pipe": pipe, "X_cols": X_cols}, f"{path_prefix}.joblib")
    # Also export to JSON (weights + scaler) if you plan MCU deployment:
    scaler: RobustScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    export = {
        "X_cols": X_cols,
        "scaler": {
            "center_": scaler.center_.tolist() if hasattr(scaler, "center_") else None,
            "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            "quantile_range": list(scaler.quantile_range),
        },
        "clf": {
            "coef_": clf.coef_.tolist(),  # shape (3, D)
            "intercept_": clf.intercept_.tolist(),  # shape (3,)
            "classes_": clf.classes_.tolist(),
        }
    }
    with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

def grid_search_hysteresis(pipe, X_cols, df_val):
    X = df_val[X_cols].values.astype(np.float32)
    y = df_val["y"].values.astype(int)
    P = pipe.predict_proba(X)
    flow = df_val["CrossPerMin"].values.astype(float)

    best = None
    for enter_med, exit_low, enter_high, exit_med in product(
        np.linspace(0.50, 0.70, 5),   # enter Medium
        np.linspace(0.35, 0.55, 5),   # exit to Low
        np.linspace(0.55, 0.80, 6),   # enter High  (search lower than 0.75!)
        np.linspace(0.50, 0.70, 5)    # exit to Med
    ):
        # simple flow-aware tweak: relax if crossings/min >= 4
        adj = (flow >= 4.0).astype(float) * 0.10
        N = len(P)
        state = np.zeros(N, dtype=int)
        for i in range(N):
            p_low, p_med, p_high = P[i]
            eM, eH = enter_med - adj[i], enter_high - adj[i]
            if state[i-1] if i>0 else 0 == 0:
                state[i] = 1 if (p_med + p_high) >= eM else 0
            elif state[i-1] == 1:
                if p_high >= eH: state[i] = 2
                elif (p_med + p_high) <= exit_low: state[i] = 0
                else: state[i] = 1
            else:
                state[i] = 1 if p_high <= exit_med else 2

        f1m = f1_score(y, state, average="macro")
        if best is None or f1m > best["f1"]:
            best = {"f1": f1m, "params": (enter_med, exit_low, enter_high, exit_med)}
    return best

# =========================
# 9) MAIN
# =========================
def main():
    # 9.1 Load feature arrays you computed previously
    feat = load_feature_arrays()
    win_centers = feat["win_centers_sec"]

    # 9.2 Load your per-second labels (time, tier per sec)
    # ---------- YOUR_DATA_HERE ----------
    # Example placeholders — REPLACE with your real label timeline:
    label_times_sec = head_count_df['Time to start'].copy()         # e.g., [0,1,2,...]
    label_tiers_sec = head_count_df['Tiers']        # e.g., ["Low","Low","Med",...]

    # raise NotImplementedError("Wire your 1 Hz label time series here")

    # 9.3 Align to windows
    y_win = align_labels_to_windows(win_centers, label_times_sec, label_tiers_sec, win_sec=WIN_SEC)

    # 9.4 Build a clean dataframe
    df_all = build_dataframe(feat, y_win)
    # print(df_all['CrossPerMin'].mean())
    # print(df_all.loc[df_all["CrossPerMin"] >= 1.0, ["t_center", "CrossPerMin", "y"]])

    # 9.5 Time-aware split (first 70% train, last 30% val)
    df_train, df_val = time_holdout_split(df_all, holdout_frac=0.30)

    # 9.6 Train
    pipe, X_cols = train_temporal_logreg(df_all)

    # best = grid_search_hysteresis(pipe, X_cols, df_train)
    # print(best)

    # 9.7 Evaluate
    ev_train = evaluate(pipe, X_cols, df_all, method="ema")
    # ev_val   = evaluate(pipe, X_cols, df_val,   use_hysteresis=True)


    print("\nTRAIN:\n", ev_train["report"])
    print("Confusion (train):", ev_train["confusion_matrix"], "Macro-F1:", ev_train["macro_f1"])
    # print("\nVAL:\n", ev_val["report"])
    # print("Confusion (val):", ev_val["confusion_matrix"], "Macro-F1:", ev_val["macro_f1"])

    # 9.8 Save model
    # save_pipeline(pipe, X_cols, path_prefix="temporal_logreg_crowdedness")

    def diag(pipe, X_cols, df):
        X = df[X_cols].values.astype(np.float32)
        y = df["y"].values.astype(int)
        P = pipe.predict_proba(X)
        yhat = np.argmax(P, axis=1)

        print("Counts y:", np.bincount(y, minlength=3))
        print("Counts yhat:", np.bincount(yhat, minlength=3))
        print("Mean P (Low,Med,High):", np.round(P.mean(0), 3))
        print(classification_report(y, yhat, target_names=["Low", "Medium", "High"]))
        print("Confusion:\n", confusion_matrix(y, yhat, labels=[0, 1, 2]))

    # diag(pipe, X_cols, df_train)

if __name__ == "__main__":
    main()

