import os
import argparse
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from eth_hash.auto import keccak
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from itertools import combinations
from scipy.stats import norm

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "dejavuserif"





# ---------------------------
# Config
# ---------------------------

# UBFC root directory
# UBFC_NPZ_ROOT is provided through --input_dir at runtime.

FRAMES_KEY = "video"

# how many seconds to use
SECONDS_USED = 30

TEMPLATE_DIM = 64

# rPPG freq
LOW_HZ = 0.7
HIGH_HZ = 4.0

DEFAULT_FPS = 30.0

# ---------------------------
# functions
# ---------------------------


def fuse_templates(templates_int16, method="median"):
    """
    Fuse multiple int16 templates into one int16 template.
    method: "median" (recommended) or "mean"
    """
    arr = np.stack([t.astype(np.float32) for t in templates_int16], axis=0)
    if method == "median":
        fused = np.median(arr, axis=0)
    elif method == "mean":
        fused = np.mean(arr, axis=0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    return np.round(fused).astype(np.int16)


def aggregate_scores(scores, method="median"):
    """
    Aggregate multiple probe-window distances into a single decision score.
    method: "min" (security-favoring), "median" (robust), "mean"
    """
    s = np.asarray(scores, dtype=np.float64)
    if method == "min":
        return float(np.min(s))
    if method == "median":
        return float(np.median(s))
    if method == "mean":
        return float(np.mean(s))
    raise ValueError(f"Unknown score aggregation method: {method}")


def compute_verification_scores(
    npz_path,
    subject_id,
    total_sec=30,
    win_sec=15,
    stride_sec=5,
    use_hamming=True,
    enroll_windows=(0.0, 5.0, 10.0),
    enroll_win_sec=None,
    enroll_fusion="median",     # NEW: "median"
    probe_agg="median",         # NEW: aggregate multiple probe distances
    probe_group_k=3,            # NEW: group K consecutive windows into one decision
):
    """
    ENROLL-vs-PROBE genuine scores with:
      - enrollment fusion via median (recommended)
      - probe multi-window aggregation: group K consecutive windows and aggregate their distances

    Returns:
      genuine_scores: list[float]  (one score per probe group)
      enroll_pair: (subject_id, enroll_template_int16)
    """
    enroll_win_sec = win_sec if enroll_win_sec is None else enroll_win_sec

    # -------- Enrollment templates (multiple windows -> fused) --------
    enroll_tpls = []
    for st in enroll_windows:
        _, pos_filt, fs = extract_pos_from_npz(
            npz_path,
            fps=DEFAULT_FPS,
            start_sec=float(st),
            duration_sec=float(enroll_win_sec),
        )
        tpl = build_template_from_signal(pos_filt, fs, dim=TEMPLATE_DIM)
        enroll_tpls.append(tpl)

    enroll_tpl = fuse_templates(enroll_tpls, method=enroll_fusion)

    # -------- Probe windows -> distances --------
    starts = get_window_starts(total_sec=total_sec, win_sec=win_sec, stride_sec=stride_sec)
    probe_dists = []
    for st in starts:
        _, pos_filt, fs = extract_pos_from_npz(
            npz_path,
            fps=DEFAULT_FPS,
            start_sec=float(st),
            duration_sec=float(win_sec),
        )
        probe_tpl = build_template_from_signal(pos_filt, fs, dim=TEMPLATE_DIM)

        if use_hamming:
            d = template_distance_hamming_sign(enroll_tpl, probe_tpl)
        else:
            d = template_distance_l2(enroll_tpl, probe_tpl)
        probe_dists.append(float(d))

    # -------- Group K consecutive windows -> one aggregated score --------
    genuine_scores = []
    k = int(probe_group_k)
    if k <= 1:
        # no grouping, each window is one score
        genuine_scores = probe_dists
    else:
        for i in range(0, len(probe_dists) - k + 1):
            group = probe_dists[i:i+k]
            genuine_scores.append(aggregate_scores(group, method=probe_agg))

    return genuine_scores, (subject_id, enroll_tpl)


def report_frr_at_far(genuine_scores, impostor_scores, target_far=0.01, num_thresholds=2000):
    """
    Return (t, FAR, FRR) where FAR <= target_far and FRR is reported at that operating point.
    Uses the smallest threshold achieving FAR <= target_far (security-favoring).
    """
    g = np.asarray(genuine_scores, dtype=np.float64)
    i = np.asarray(impostor_scores, dtype=np.float64)

    all_scores = np.concatenate([g, i])
    thresholds = np.linspace(float(all_scores.min()), float(all_scores.max()), num_thresholds)

    FAR = np.array([np.mean(i <= t) for t in thresholds], dtype=np.float64)
    FRR = np.array([np.mean(g > t) for t in thresholds], dtype=np.float64)

    idx = np.where(FAR <= target_far)[0]
    if len(idx) == 0:
        j = int(np.argmin(FAR))
        return float(thresholds[j]), float(FAR[j]), float(FRR[j])

    j = int(idx[-1])  # first threshold achieving FAR <= target
    return float(thresholds[j]), float(FAR[j]), float(FRR[j])


def list_npz_files(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))


def parse_subject_id(filepath):
    base = os.path.basename(filepath)
    name, _ = os.path.splitext(base)
    return name


def bandpass_filter(signal, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ, order=3):
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def load_npz_frames(npz_path):
    data = np.load(npz_path,allow_pickle=True)
    if FRAMES_KEY not in data:
        raise KeyError(f"Key '{FRAMES_KEY}' not found in {npz_path}. "
                       f"Available keys: {list(data.keys())}")
    frames = data[FRAMES_KEY]  # expected shape: (T, H, W, 3)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected frames shape {frames.shape} in {npz_path}")
    return frames  # float or uint8


def extract_pos_from_npz(npz_path, fps=DEFAULT_FPS, start_sec=0.0, duration_sec=SECONDS_USED):
    """
    POS
    Split window：[start_sec, start_sec+duration_sec)
    return：pos_raw, pos_filt, fps
    """
    frames = load_npz_frames(npz_path)  # (T, H, W, 3)
    T = frames.shape[0]

    start = int(start_sec * fps)
    end = int((start_sec + duration_sec) * fps)

    start = max(0, min(T, start))
    end = max(0, min(T, end))

    frames = frames[start:end]
    T_used = frames.shape[0]
    if T_used < 10:
        raise RuntimeError(f"Too few frames after slicing in {npz_path} (used={T_used})")

    rgb_traces = frames.reshape(T_used, -1, 3).mean(axis=1)  # (T, 3)

    rgb_norm = (rgb_traces - rgb_traces.mean(axis=0)) / (rgb_traces.std(axis=0) + 1e-8)
    R = rgb_norm[:, 0]
    G = rgb_norm[:, 1]
    B = rgb_norm[:, 2]

    X = R - G
    Y = R + G - 2 * B
    X = X - X.mean()
    Y = Y - Y.mean()

    alpha = np.std(X) / (np.std(Y) + 1e-8)
    pos_raw = X - alpha * Y

    pos_filt = bandpass_filter(pos_raw, fs=fps, low_hz=LOW_HZ, high_hz=HIGH_HZ)
    return pos_raw, pos_filt, fps


def build_template_from_signal(signal, fs, dim=TEMPLATE_DIM):
    """
    Use PSD to build fixed-lenght template
    - Use Welch to calculate PSD
    - limit to heart freq
    - interplot to dim
    - log + normalization + quantify
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    freqs_band = freqs[mask]
    psd_band = psd[mask]

    if len(psd_band) < 4:
        psd_band = np.pad(psd_band, (0, max(0, 4 - len(psd_band))), mode="edge")
        freqs_band = np.linspace(LOW_HZ, HIGH_HZ, len(psd_band))

    target_freqs = np.linspace(LOW_HZ, HIGH_HZ, dim)
    psd_interp = np.interp(target_freqs, freqs_band, psd_band)

    eps = 1e-10
    feat = np.log(psd_interp + eps)
    feat = (feat - feat.mean()) / (feat.std() + 1e-8)

    feat_quant = np.round(feat * 1000).astype(np.int16)
    return feat_quant


def template_to_keccak_hash(template_vec):
    """
    template → bytes → keccak256 → 0x hex string
    """
    template_bytes = template_vec.tobytes()
    digest = keccak(template_bytes)  # 32 bytes
    hex_str = "0x" + digest.hex()
    
    return hex_str


def get_window_starts(total_sec=30, win_sec=15, stride_sec=5):
    starts = []
    s = 0
    while s + win_sec <= total_sec + 1e-9:
        starts.append(float(s))
        s += stride_sec
    return starts

def template_distance_l2(a, b):
    # normalization L2
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-8))

def template_distance_hamming_sign(a, b):
    # hamming: sign-binarization (a>0)
    bits_a = (a > 0).astype(np.uint8)
    bits_b = (b > 0).astype(np.uint8)
    return float((bits_a != bits_b).mean())

def compute_within_session_distances(npz_path, subject_id,
                                     total_sec=30, win_sec=15, stride_sec=5,
                                     use_hamming=False):
    """
    slide window template from the same video
    return list[dict]，every dict is the len of a window paris
    """
    starts = get_window_starts(total_sec=total_sec, win_sec=win_sec, stride_sec=stride_sec)

    templates = []
    for st in starts:
        _, pos_filt, fs = extract_pos_from_npz(npz_path, fps=DEFAULT_FPS, start_sec=st, duration_sec=win_sec)
        tpl = build_template_from_signal(pos_filt, fs, dim=TEMPLATE_DIM)
        templates.append((st, st + win_sec, tpl))

    records = []
    for (s1, e1, t1), (s2, e2, t2) in combinations(templates, 2):
        if use_hamming:
            d = template_distance_hamming_sign(t1, t2)
            metric = "hamming_sign"
        else:
            d = template_distance_l2(t1, t2)
            metric = "l2_norm"

        records.append({
            "subject_id": subject_id,
            "npz_path": npz_path,
            "metric": metric,
            "win_sec": win_sec,
            "stride_sec": stride_sec,
            "seg1": f"{int(s1)}-{int(e1)}",
            "seg2": f"{int(s2)}-{int(e2)}",
            "distance": d
        })
    return records

def plot_distance_hist(distances, out_path="within_session_distance_hist.pdf", metric_name="l2_norm"):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.5, 3.2))
    plt.hist(distances, bins=15, density=True)
    plt.xlabel(f"Within-session template distance ({metric_name})")
    plt.ylabel("Density")
    plt.title("Within-session template stability (sliding windows)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram to {out_path}")

def plot_raw_psd_template_triplet(pos_raw, pos_filt, fs, template_vec,
                                  out_path="psd_triplet.pdf"):
    """
    3-stage plots：
      (a) raw POS rPPG (time domain)
      (b) band-limited Welch PSD
      (c) final quantized template (int16[64], full-width)
    """
    # Welch PSD (on filtered signal)
    freqs, psd = welch(pos_filt, fs=fs, nperseg=min(256, len(pos_filt)))
    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    freqs_band = freqs[mask]
    psd_band = psd[mask]

    # ---- layout: 2 rows, 2 cols ----
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        height_ratios=[1, 1.1],
        hspace=0.6, wspace=0.3
    )

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_psd = fig.add_subplot(gs[0, 1])
    ax_tpl = fig.add_subplot(gs[1, :])  # full width

    # (a) Raw rPPG
    t = np.arange(len(pos_raw)) / fs
    ax_raw.plot(t, pos_raw, linewidth=1)
    ax_raw.set_title("(a) Raw POS rPPG (time domain)")
    ax_raw.set_xlabel("Time (s)")
    ax_raw.set_ylabel("Amplitude (a.u.)")
    ax_raw.grid(True, alpha=0.3)

    # (b) Filtered PSD
    ax_psd.semilogy(freqs_band, psd_band, linewidth=1.5)
    ax_psd.axvspan(LOW_HZ, HIGH_HZ, alpha=0.15)
    ax_psd.set_title("(b) Band-limited Welch PSD")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power")
    ax_psd.set_xlim(LOW_HZ, HIGH_HZ)
    ax_psd.grid(True, alpha=0.3)

    # (c) Quantized template (full width)
    ax_tpl.bar(np.arange(len(template_vec)), template_vec, width=0.8)
    ax_tpl.set_title("(c) Quantized 64-D Physiological Template (int16)")
    ax_tpl.set_xlabel("Template index")
    ax_tpl.set_ylabel("Value")
    ax_tpl.grid(True, alpha=0.3)


    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved triplet figure to {out_path}")


def compute_inter_user_distances(subject_templates, use_hamming=True):
    """
    Inter-user (impostor) distances.
    subject_templates: list of (subject_id, template_vec)
    Returns: list[float]
    """
    records = []
    print(f"Computing inter-user distances for {len(subject_templates)} subjects...")

    for (id1, t1), (id2, t2) in combinations(subject_templates, 2):
        if id1 == id2:
            continue

        if use_hamming:
            d = template_distance_hamming_sign(t1, t2)
        else:
            d = template_distance_l2(t1, t2)

        records.append(d)

    return records


def plot_dual_dist_hist(intra_dists, inter_dists,
                        out_path="dual_dist_hist.pdf",
                        metric_name="hamming_sign",
                        threshold=None):
    """
    Plots intra-session (genuine stability) and inter-user (impostor) together.
    """
    fig = plt.figure(figsize=(6.5, 4))

    plt.hist(intra_dists, bins=20, density=True, alpha=0.6, label='Intra-session (Stability)')
    plt.hist(inter_dists, bins=20, density=True, alpha=0.6, label='Inter-user (Security)')

    # if threshold is not None:
    #     plt.axvline(x=float(threshold), color='black', linestyle='--',
    #                 label=f'Suggested Threshold (t={threshold:.3f})')

    plt.xlabel(f"Template distance ({metric_name})")
    plt.ylabel("Density")
    plt.title("Biometric Discriminability Analysis")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dual histogram to {out_path}")



def compute_far_frr(genuine_scores, impostor_scores, thresholds):
    """
    Decision rule: ACCEPT if distance <= t
    genuine (same user) should be accepted
    impostor (different users) should be rejected
    """
    g = np.asarray(genuine_scores, dtype=np.float64)
    i = np.asarray(impostor_scores, dtype=np.float64)

    FAR = np.empty_like(thresholds, dtype=np.float64)  # impostor accepted
    FRR = np.empty_like(thresholds, dtype=np.float64)  # genuine rejected

    for k, t in enumerate(thresholds):
        FAR[k] = np.mean(i <= t)
        FRR[k] = np.mean(g > t)

    return FAR, FRR

def compute_eer(genuine_scores, impostor_scores, num_thresholds=800):
    g = np.asarray(genuine_scores, dtype=np.float64)
    i = np.asarray(impostor_scores, dtype=np.float64)

    # Build thresholds from score range (robust)
    all_scores = np.concatenate([g, i])
    t_min, t_max = float(np.min(all_scores)), float(np.max(all_scores))
    if t_max <= t_min + 1e-12:
        raise ValueError("Scores have near-zero range; cannot compute EER meaningfully.")

    thresholds = np.linspace(t_min, t_max, num_thresholds)
    FAR, FRR = compute_far_frr(g, i, thresholds)

    idx = int(np.argmin(np.abs(FAR - FRR)))
    eer = 0.5 * (FAR[idx] + FRR[idx])
    t_eer = thresholds[idx]
    return float(eer), float(t_eer), FAR, FRR, thresholds

def plot_far_frr_curve(FAR, FRR, thresholds, out_path="far_frr_curve.pdf", title="Verification Trade-off (FAR/FRR) - L2 Distance",mark_eer=True):
    FAR = np.asarray(FAR)
    FRR = np.asarray(FRR)
    thresholds = np.asarray(thresholds)

    # sort by threshold to ensure a clean curve
    idx = np.argsort(thresholds)
    thresholds, FAR, FRR = thresholds[idx], FAR[idx], FRR[idx]

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(thresholds, FAR, label="FAR (impostor accepted)")
    plt.plot(thresholds, FRR, label="FRR (genuine rejected)")

    if mark_eer:
        eer_idx = np.argmin(np.abs(FAR - FRR))
        eer_t = thresholds[eer_idx]
        eer = 0.5 * (FAR[eer_idx] + FRR[eer_idx])
        plt.scatter([eer_t], [eer], zorder=3)
        plt.annotate(f"EER={eer:.4f}", (eer_t, eer),
                     textcoords="offset points", xytext=(14, 8))

    plt.xlabel("Threshold t (accept if distance ≤ t)")
    plt.ylabel("Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved FAR/FRR curve to {out_path}")
    # plt.figure(figsize=(6.2, 4.2))
    # plt.plot(thresholds, FAR, label="FAR (impostor accepted)")
    # plt.plot(thresholds, FRR, label="FRR (genuine rejected)")
    # plt.xlabel("Threshold t (accept if distance ≤ t)")
    # plt.ylabel("Rate")
    # plt.title(title)
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(out_path, bbox_inches="tight")
    # plt.close()
    # print(f"Saved FAR/FRR curve to {out_path}")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_det_curve(FAR, FRR,
                   out_path="det_curve.pdf",
                   title="DET Curve (normal deviate scale, FRR vs FAR) – L2 distance",
                   eps=1e-6):
    """
    Strict DET curve using normal deviate (probit) axes.
    Also marks:
      - EER point
      - Operating points at FAR = 1% and 5%
    """
    mark_far = (0.01, 0.05)
    FAR = np.asarray(FAR, dtype=float)
    FRR = np.asarray(FRR, dtype=float)

    # ---- sort by FAR for clean curve ----
    order = np.argsort(FAR)
    FAR = FAR[order]
    FRR = FRR[order]

    valid = (FAR > 0) & (FAR < 1)
    FAR = FAR[valid]
    FRR = FRR[valid]

    # ---- avoid 0/1 for probit ----
    FARc = np.clip(FAR, eps, 1 - eps)
    FRRc = np.clip(FRR, eps, 1 - eps)

    x = norm.ppf(FARc)
    y = norm.ppf(FRRc)

    # ---- find EER (min |FAR - FRR|) ----
    idx_eer = int(np.argmin(np.abs(FAR - FRR)))
    eer = 0.5 * (FAR[idx_eer] + FRR[idx_eer])

    x_eer = norm.ppf(np.clip(FAR[idx_eer], eps, 1 - eps))
    y_eer = norm.ppf(np.clip(FRR[idx_eer], eps, 1 - eps))

    # ---- plot ----
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(x, y, linewidth=1.6, label="DET curve")

    # ---- EER point ----
    plt.scatter([x_eer], [y_eer], s=40, facecolors="yellow", linewidths=1.2, zorder=4, label=f"EER ≈ {eer*100:.2f}%")

    # ---- FAR = 1% / 5% operating points ----
    for tf in mark_far:
        tf_clamped = float(np.clip(tf, FAR.min(), FAR.max()))
        fr = float(np.interp(tf_clamped, FAR, FRR))

        xt = norm.ppf(np.clip(tf_clamped, eps, 1 - eps))
        yt = norm.ppf(np.clip(fr, eps, 1 - eps))

        if tf == 0.01:
            facecolor = 'orange'
        else:
            facecolor = "pink"
        plt.scatter([xt], [yt], s=36, facecolor=facecolor, zorder=4, marker="s", label=f"FAR={tf*100:.0f}%")

        # optional guide lines
        plt.axvline(xt, linestyle=":", linewidth=0.9, alpha=0.5, color="0.5", zorder=1)
        plt.axhline(yt, linestyle=":", linewidth=0.9, alpha=0.5, color="0.5", zorder=1)

    # ---- DET ticks (standard) ----
    ticks = np.array([0.001, 0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99, 0.999])
    tick_labels = [f"{t*100:.1f}%" for t in ticks]
    tick_locations = norm.ppf(ticks)

    plt.xticks(tick_locations, tick_labels)
    plt.yticks(tick_locations, tick_labels)

    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(
        f"Saved DET curve to {out_path} | "
        f"EER≈{eer*100:.2f}%, "
        f"FRR@1%≈{np.interp(0.01, FAR, FRR)*100:.2f}%, "
        f"FRR@5%≈{np.interp(0.05, FAR, FRR)*100:.2f}%"
    )


def report_operating_point(FAR, FRR, thresholds, target_far=0.01):
    idx = np.where(FAR <= target_far)[0]
    if len(idx) == 0:
        # if never reaches target FAR, pick minimum FAR point
        j = int(np.argmin(FAR))
        return thresholds[j], FAR[j], FRR[j]
    j = int(idx[-1])  # largest threshold with FAR <= target
    return thresholds[j], FAR[j], FRR[j]


# def compute_verification_scores(
#     npz_path,
#     subject_id,
#     total_sec=30,
#     win_sec=15,
#     stride_sec=5,
#     use_hamming=True,
#     enroll_windows=(0.0, 5.0, 10.0),
#     enroll_win_sec=None,
# ):
#     """
#     Compute ENROLL-vs-PROBE genuine scores for one subject.
#
#     Enrollment:
#       - average template from multiple enrollment windows
#
#     Probe:
#       - sliding windows across the whole sequence
#
#     Returns:
#       genuine_scores : list[float]
#       enroll_pair    : (subject_id, enroll_template_int16)
#     """
#     enroll_win_sec = win_sec if enroll_win_sec is None else enroll_win_sec
#
#     # -------- Enrollment templates --------
#     enroll_templates = []
#     for st in enroll_windows:
#         _, pos_filt, fs = extract_pos_from_npz(
#             npz_path,
#             fps=DEFAULT_FPS,
#             start_sec=float(st),
#             duration_sec=float(enroll_win_sec),
#         )
#         tpl = build_template_from_signal(pos_filt, fs, dim=TEMPLATE_DIM)
#         enroll_templates.append(tpl)
#
#     # Average enrollment templates (int16-safe)
#     if len(enroll_templates) == 1:
#         enroll_tpl = enroll_templates[0]
#     else:
#         arr = np.stack([t.astype(np.float32) for t in enroll_templates], axis=0)
#         enroll_tpl = np.round(arr.mean(axis=0)).astype(np.int16)
#
#     # -------- Probe templates (sliding windows) --------
#     starts = get_window_starts(
#         total_sec=total_sec,
#         win_sec=win_sec,
#         stride_sec=stride_sec,
#     )
#
#     genuine_scores = []
#     for st in starts:
#         _, pos_filt, fs = extract_pos_from_npz(
#             npz_path,
#             fps=DEFAULT_FPS,
#             start_sec=float(st),
#             duration_sec=float(win_sec),
#         )
#         probe_tpl = build_template_from_signal(pos_filt, fs, dim=TEMPLATE_DIM)
#
#         if use_hamming:
#             d = template_distance_hamming_sign(enroll_tpl, probe_tpl)
#         else:
#             d = template_distance_l2(enroll_tpl, probe_tpl)
#
#         genuine_scores.append(float(d))
#
#     return genuine_scores, (subject_id, enroll_tpl)

# ---------------------------
# main pipeline
# ---------------------------
def process_ubfc_npz(root_dir, output_csv="ubfc_npz_commitments.csv",
                     max_files=None,
                     do_within_session=True,
                     within_csv="ubfc_within_session_distances.csv",
                     use_hamming=True):

    files = list_npz_files(root_dir)
    if max_files is not None:
        files = files[:max_files]
    print(f"Found {len(files)} npz files.")

    records = []
    dist_records = []
    subject_base_templates = []  # To store one 30s template per subject for inter-user test
    genuine_scores_all = []  # list[float]
    enroll_templates = []  # list[(subject_id, enroll_template)]

    for idx, fpath in enumerate(files):
        try:
            print(f"[{idx+1}/{len(files)}] Processing {fpath}")
            subject_id = parse_subject_id(fpath)

            pos_raw, pos_filt, fs = extract_pos_from_npz(fpath, fps=DEFAULT_FPS)

            template_vec = build_template_from_signal(pos_filt, fs, dim=TEMPLATE_DIM)
            subject_base_templates.append((subject_id, template_vec))

            ## plot Welch-PSD (any frame)
            if idx == 10:
                plot_raw_psd_template_triplet(
                    pos_raw=pos_raw,
                    pos_filt=pos_filt,
                    fs=fs,
                    template_vec=template_vec,
                    out_path="../../psd_triplet.pdf")

            commitment_hex = template_to_keccak_hash(template_vec)

            records.append({
                "npz_path": fpath,
                "subject_id": subject_id,
                "fps": fs,
                "template_dim": TEMPLATE_DIM,
                "commitment_hex": commitment_hex
            })

            if do_within_session:
                dist_records.extend(
                    compute_within_session_distances(
                        npz_path=fpath,
                        subject_id=subject_id,
                        total_sec=30,
                        win_sec=15,
                        stride_sec=5,
                        use_hamming=use_hamming
                    )
                )

            # ---- verification: enroll-vs-probe genuine scores ----
            g_scores, enroll_pair = compute_verification_scores(
                npz_path=fpath,
                subject_id=subject_id,
                total_sec=30,
                win_sec=15,
                stride_sec=5,
                use_hamming=use_hamming,
                enroll_windows=(0.0, 5.0, 10.0),  # 你也可以加到 (0,5,10,15,20)
                enroll_fusion="median",  # NEW
                probe_agg="median",  # NEW: "min"/"median"/"mean"
                probe_group_k=3  # NEW: 3-window aggregation
            )
            genuine_scores_all.extend(g_scores)
            enroll_templates.append(enroll_pair)



        except Exception as e:
            print(f"⚠️ Error processing {fpath}: {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} records to {output_csv}")
    intra_distances = []
    if do_within_session:
        ddf = pd.DataFrame(dist_records)
        ddf.to_csv(within_csv, index=False)
        intra_distances = ddf["distance"].values
        print(f"Saved {len(ddf)} distance pairs to {within_csv}")

        # plot
        plot_distance_hist(
            distances=ddf["distance"].values,
            out_path="within_session_distance_hist.pdf",
            metric_name=ddf["metric"].iloc[0] if len(ddf) else ("hamming_sign" if use_hamming else "l2_norm")
        )

    inter_distances = compute_inter_user_distances(subject_base_templates, use_hamming=use_hamming)
    metric_label = "hamming_sign" if use_hamming else "l2_norm"
    print("Intra metric:", ("hamming_sign" if use_hamming else "l2_norm"),
          "range:", np.min(intra_distances), np.max(intra_distances))
    print("Inter metric:", ("hamming_sign" if use_hamming else "l2_norm"),
          "range:", np.min(inter_distances), np.max(inter_distances))

    plot_dual_dist_hist(intra_distances, inter_distances,
                        out_path=f"security_analysis_hist_{metric_label}.pdf",
                        metric_name=metric_label, threshold=0.35)



    # --- EER + FAR/FRR curves ---

    # =========================
    # Verification evaluation
    # =========================

    impostor_scores = compute_inter_user_distances(
        enroll_templates,
        use_hamming=use_hamming
    )

    n_impostor = len(impostor_scores)
    n_genuine = len(genuine_scores_all)
    eps = 0.5 / max(n_impostor, n_genuine)

    eer, t_eer, FAR, FRR, thresholds = compute_eer(
        genuine_scores_all,
        impostor_scores,
        num_thresholds=1200
    )

    print(f"[Verification] EER = {eer:.4f} at threshold t = {t_eer:.4f}")

    metric_label = "hamming_sign" if use_hamming else "L2 Distance"

    plot_far_frr_curve(
        FAR, FRR, thresholds,
        out_path=f"far_frr_curve_{metric_label}.pdf",
        title=f"Verification Trade-off (FAR/FRR) - {metric_label},",
        mark_eer=True
    )

    plot_det_curve(
        FAR, FRR,
        out_path=f"det_curve_{metric_label}.pdf",
        title=f"DET Curve (normal deviate scale, FRR vs FAR) - {metric_label}",
        eps=eps
    )

    t_1pct, far_1pct, frr_1pct = report_operating_point(
        FAR, FRR, thresholds, target_far=0.01
    )
    t1, far1, frr1 = report_frr_at_far(genuine_scores_all, impostor_scores, target_far=0.01)
    t5, far5, frr5 = report_frr_at_far(genuine_scores_all, impostor_scores, target_far=0.05)
    print(
        f"[OpPoint] FAR<=1% achieved at t={t_1pct:.4f}: "
        f"FAR={far_1pct:.4f}, FRR={frr_1pct:.4f}"
    )



    print(f"[OpPoint] FRR@FAR<=1%:  t={t1:.4f}, FAR={far1:.4f}, FRR={frr1:.4f}")
    print(f"[OpPoint] FRR@FAR<=5%:  t={t5:.4f}, FAR={far5:.4f}, FRR={frr5:.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate BARS rPPG template commitments from UBFC npz files."
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Directory containing UBFC npz files."
    )
    parser.add_argument(
        "-o", "--output_csv",
        default="ubfc_npz_commitments.csv",
        help="Output CSV file for template commitments."
    )
    parser.add_argument(
        "--within_csv",
        default="ubfc_within_session_distances.csv",
        help="Output CSV file for within-session distances."
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional maximum number of npz files to process."
    )
    parser.add_argument(
        "--no_within_session",
        action="store_true",
        help="Disable within-session distance computation."
    )
    parser.add_argument(
        "--use_hamming",
        action="store_true",
        help="Use sign-binarized Hamming distance instead of L2."
    )
    args = parser.parse_args()

    df = process_ubfc_npz(
        args.input_dir,
        output_csv=args.output_csv,
        max_files=args.max_files,
        do_within_session=not args.no_within_session,
        within_csv=args.within_csv,
        use_hamming=args.use_hamming
    )
    print(df.head())