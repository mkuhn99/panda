"""Fine-tune (or train from scratch) Panda on the EEG channels of one or more BIDS EEG datasets.

Datasets are BIDS folders under --data-root, named on the command line (default: nm000111 nm000287 on004902
on005555 on006695; nm000287 is the Muse sleep-onset set). Only channels typed EEG in the BIDS channels.tsv are used
(EOG / EMG / ECG / MISC / ... and channels marked bad are dropped); without a channels.tsv, mne's EEG types are used.

Stages (Panda needs transformers==4.40.2, numpy<2; prep needs mne):
  scan   lists each dataset's EEG files, formats, EEG channel counts and splits from the BIDS sidecars, without
         loading any signal. Run it first on a new machine.
  prep   each recording -> EEG channels, 128 Hz, 0.5-40 Hz, uV, Hann 15 (as in panda_muse_clean_eval.py), saved as
         <cache-dir>/<dataset>_hann<k>/<rec>.npy (T, ch) plus a per-channel 1 s bad-epoch mask (n_epochs, ch) by the
         artifact rule of dynamix_muse_clean_eval.py, and an index.csv. --jobs N prepares recordings in parallel.
  train  epochs over non-overlapping (context 512 + horizon 128) windows of every recording, with a new random
         offset each epoch; each window is standardized with its context statistics (as in the eval), MSE on the
         next 128 samples. If a cache is missing, prep runs in-process when mne is importable, otherwise through
         --prep-python.

Channels: recordings of a dataset are pooled by channel count, and each batch comes from one pool (Panda's channel
attention needs equal channel counts within a batch). Windows of recordings with more than --max-channels EEG
channels get a random subset of --max-channels channels each time they are drawn (kept in montage order).
Artifacts: --clean (all datasets) or --clean DS ... (only those) keeps only windows where at least
min(n_ch, --max-channels) channels have no bad 1 s epoch, and draws the channels from those clean ones. For 4-channel
Muse this is the earlier rule (all 4 channels clean). The 100 uV peak-to-peak limit also rejects deep-sleep slow
waves, so on sleep PSG datasets --clean drops much of N3.
Splits: recordings whose BIDS sessions.tsv says split=test are never used (Muse test split stays for the eval).
--weights scales how many of a dataset's windows go into an epoch (0.5 = a random half, 2 = each window twice), so
long overnight recordings can be kept from swamping the short ones.
Validation: --val-frac of each dataset's train subjects are held out (--val-datasets to limit this); MSE and MAE over
the first 120 steps (the eval's n-step MAE) on --val-windows fixed windows per dataset, before training (zero-shot)
and after every epoch, over all val data (val/mse, val/mae_n120) and per dataset (val/<ds>/mse, val/<ds>/mae_n120;
nm000287 = Muse), next to the predict-zero baseline (val/zero_mse, val/<ds>/zero_mse). Train MSE is logged per dataset
too (train/mse_<ds>, train/epoch_mse_<ds>). The weights of the epoch with the lowest --select-metric (default the Muse
val MSE, as in the earlier Muse-only runs) go to checkpoint-best; training stops early after --patience epochs without
improvement.
Outputs per run: checkpoint-best, checkpoint-final, log.csv (train / val loss, lr, grad norm per step and epoch),
summary.json (best epoch, early stop). With --wandb the same statistics go
to Weights & Biases.

Checkpoints are saved with save_pretrained, so they plug straight into the existing eval scripts:
    python scripts/panda/train_muse.py scan --data-root /workspace/datasets
    python scripts/panda/train_muse.py prep --data-root /workspace/datasets --cache-dir /workspace/cache_dir/panda_train --jobs 8
    python scripts/panda/train_muse.py train --data-root /workspace/datasets --cache-dir /workspace/cache_dir/panda_train \
        --batch-size 256 --lr 3e-4 --epochs 30 --patience 4 --clean nm000287 --weights 1 0.3 1 0.3 1
    python scripts/panda/train_muse.py train --datasets nm000287 --clean --data-root /workspace/datasets   # Muse only
    python ../panda_muse_clean_eval.py --model ../save_dir/panda_muse/<run>/checkpoint-best
"""
import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # panda repo
WORK = os.path.dirname(REPO)  # folder holding the panda repo (defaults only; all paths are arguments)
MODEL = "GilpinLab/panda"
SFREQ = 128
N_MAE = 120
DATASETS = ["nm000111", "nm000287", "on004902", "on005555", "on006695"]
MUSE = "nm000287"
EEG_EXT = (".vhdr", ".edf", ".bdf", ".set", ".fif", ".cnt", ".mff")  # order = preference if a recording has several
P2P_UV, FLAT_UV, Z_MAX = 100.0, 0.5, 3.0  # artifact rule of dynamix_muse_clean_eval.py


def cache_dir(args, dataset):
    return os.path.join(os.path.expanduser(args.cache_dir), f"{dataset}_hann{args.hann}")


def dataset_root(args, dataset):
    return os.path.join(os.path.expanduser(args.data_root), dataset)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


# ---------------------------------------------------------------- BIDS (no mne needed)

def read_tsv(f):
    with open(f, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def entities(path):
    """BIDS key-value entities of a file name, e.g. {'sub': '01', 'ses': '1', 'task': 'rest'}."""
    out = {}
    for part in os.path.basename(path).split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            out[k] = v
    return out


def eeg_files(root):
    """One EEG file per recording (by BIDS stem), preferring formats in EEG_EXT order."""
    best = {}
    for ext in EEG_EXT:
        for f in glob.glob(os.path.join(root, "sub-*", "**", "eeg", f"*_eeg{ext}"), recursive=True):
            best.setdefault(f[:-len(ext)], f)
    return sorted(best.values())


def channels_tsv(f):
    """The channels.tsv that applies to an EEG file: same stem, else the one in its folder (or a parent's, by BIDS
    inheritance) whose entities are a subset of the file's."""
    stem = f[:f.rindex("_eeg")]
    if os.path.exists(stem + "_channels.tsv"):
        return stem + "_channels.tsv"
    ent = entities(stem)
    d = os.path.dirname(f)
    while True:
        cands = [c for c in glob.glob(os.path.join(d, "*_channels.tsv"))
                 if all(ent.get(k) == v for k, v in entities(c[:-len("_channels.tsv")]).items())]
        if cands:
            return max(cands, key=lambda c: len(entities(c)))  # the most specific one
        if os.path.exists(os.path.join(d, "dataset_description.json")) or os.path.dirname(d) == d:
            return None
        d = os.path.dirname(d)


def eeg_channels(f):
    """Names of the good EEG channels of a recording from its channels.tsv; None if there is no channels.tsv."""
    tsv = channels_tsv(f)
    if tsv is None:
        return None
    rows = read_tsv(tsv)
    return [r["name"] for r in rows if r.get("type", "").strip().upper() == "EEG"
            and r.get("status", "good").strip().lower() != "bad"]


def sessions(root):
    """{(sub, ses): sessions.tsv row} over the dataset (split / n2_onset columns if present)."""
    out = {}
    for f in glob.glob(os.path.join(root, "sub-*", "sub-*_sessions.tsv")):
        sub = os.path.basename(f).split("_")[0][4:]
        for r in read_tsv(f):
            out[(sub, r.get("session_id", "")[4:])] = r
    return out


def recordings(root):
    """(name, file, meta) per EEG recording of a BIDS dataset."""
    ses_rows = sessions(root)
    for f in eeg_files(root):
        ent = entities(f)
        sub, ses = ent.get("sub", ""), ent.get("ses", "")
        row = ses_rows.get((sub, ses), {})
        name = os.path.basename(f[:f.rindex("_eeg")])
        yield name, f, dict(sub=sub, ses=ses, split=(row.get("split") or "train").strip(),
                            n2_onset=row.get("n2_onset", ""))


def scan(args, dataset):
    root = dataset_root(args, dataset)
    recs = list(recordings(root))
    if not recs:
        print(f"{dataset}: no sub-*/**/eeg/*_eeg.* files under {root}\n", flush=True)
        return
    fmt = Counter(os.path.splitext(f)[1] for _, f, _ in recs)
    n_ch, types, missing = Counter(), Counter(), 0
    for _, f, _ in recs:
        tsv = channels_tsv(f)
        if tsv is None:
            missing += 1
            continue
        rows = read_tsv(tsv)
        types.update(r.get("type", "?").strip().upper() for r in rows)
        n_ch[len(eeg_channels(f))] += 1
    print(f"{dataset}: {len(recs)} recordings of {len({m['sub'] for _, _, m in recs})} subjects under {root}")
    print(f"  formats {dict(fmt)}; splits {dict(Counter(m['split'] for _, _, m in recs))}")
    print(f"  channel types over all recordings {dict(types)}")
    print(f"  EEG channels per recording {dict(sorted(n_ch.items()))}"
          + (f"; {missing} without channels.tsv (mne's EEG types will be used)" if missing else ""))
    first = next((f for _, f, _ in recs if channels_tsv(f)), None)
    if first:
        print(f"  e.g. {os.path.basename(first)}: {eeg_channels(first)}")
    print(flush=True)


# ---------------------------------------------------------------- prep (env with mne)

def smooth(x, k):
    if k <= 1:
        return x
    w = np.hanning(k + 2)[1:-1]
    w /= w.sum()
    return np.stack([np.convolve(c, w, mode="same") for c in x.T], axis=1)


def robust_z(a):
    med = np.median(a, 0)
    mad = np.median(np.abs(a - med), 0) * 1.4826 + 1e-12
    return (a - med) / mad


def bad_epochs(x):
    """(n_epochs, ch) bool for 1 s epochs of the 0.5-40 Hz signal x (T, ch) in uV: peak-to-peak, flat, or robust
    outlier in log variance / log 20-40 Hz power, per channel."""
    from scipy.signal import welch
    n = len(x) // SFREQ
    ep = x[:n * SFREQ].reshape(n, SFREQ, -1)
    sd = ep.std(1)
    f, p = welch(ep, fs=SFREQ, nperseg=SFREQ, axis=1)
    emg = p[:, (f >= 20) & (f <= 40)].sum(1)
    bad = (np.ptp(ep, axis=1) > P2P_UV) | (sd < FLAT_UV)
    bad |= robust_z(np.log(sd + 1e-12)) > Z_MAX
    bad |= robust_z(np.log(emg + 1e-12)) > Z_MAX
    return bad


def read_raw(f, channels):
    """EEG channels of a recording, 128 Hz, 0.5-40 Hz, (T, ch) float32 in uV, and the channel names."""
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw(f, preload=False)
    if channels is None:
        channels = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude="bads")]
    else:
        lookup = {c.strip().strip(".").lower(): c for c in raw.ch_names}  # eegmmidb-style padded labels
        channels = [lookup[c.strip().strip(".").lower()] for c in channels if c.strip().strip(".").lower() in lookup]
    if not channels:
        raise ValueError("no EEG channels")
    raw.pick(channels)
    raw.load_data()
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ)  # mne low-passes before decimating
    raw.filter(0.5, 40.0)
    return (raw.get_data().T * 1e6).astype(np.float32), channels


def prep_one(job):
    name, f, out, hann = job
    fx, fb = os.path.join(out, f"{name}.npy"), os.path.join(out, f"{name}_bad.npy")
    fc = os.path.join(out, f"{name}_channels.json")
    try:
        if not (os.path.exists(fx) and os.path.exists(fb) and os.path.exists(fc)):
            x, chs = read_raw(f, eeg_channels(f))
            np.save(fb, bad_epochs(x))  # on the unsmoothed signal, as in the eval
            np.save(fx, smooth(x, hann).astype(np.float32))
            with open(fc, "w") as fh:
                json.dump(chs, fh)
        x = np.load(fx, mmap_mode="r")
        return name, x.shape, json.load(open(fc)), None
    except Exception as e:  # one unreadable file shouldn't stop a dataset
        return name, None, None, f"{type(e).__name__}: {e}"


def prep(args, dataset):
    root = dataset_root(args, dataset)
    recs = list(recordings(root))
    if not recs:
        raise SystemExit(f"no {dataset} recordings found under {root}")
    out = cache_dir(args, dataset)
    os.makedirs(out, exist_ok=True)
    meta = {name: m for name, _, m in recs}
    jobs = [(name, f, out, args.hann) for name, f, _ in recs]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs) as ex:
            results = list(ex.map(prep_one, jobs, chunksize=1))
    else:
        results = map(prep_one, jobs)
    rows, failed = [], 0
    for k, (name, shape, chs, err) in enumerate(results, 1):
        if err:
            failed += 1
            print(f"{dataset}: [{k}/{len(jobs)}] {name} skipped ({err})", flush=True)
            continue
        rows.append(dict(name=name, **meta[name], n_samples=shape[0], n_channels=shape[1], channels="|".join(chs)))
        print(f"{dataset}: [{k}/{len(jobs)}] {name} {shape}", flush=True)
    if not rows:
        raise SystemExit(f"{dataset}: no recording could be read")
    with open(os.path.join(out, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "source.json"), "w") as fh:
        json.dump(dict(root=root, hann=args.hann), fh)
    hours = sum(r["n_samples"] for r in rows) / SFREQ / 3600
    print(f"{dataset}: {len(rows)} recordings ({failed} skipped), {hours:.1f} h, EEG channels "
          f"{dict(Counter(r['n_channels'] for r in rows))} -> {out}", flush=True)


# ---------------------------------------------------------------- train (panda env)

def read_index(args, dataset):
    out = cache_dir(args, dataset)
    f = os.path.join(out, "index.csv")
    if not os.path.exists(f):
        try:
            import mne  # noqa: F401
            print(f"no cache for {dataset} in {out}, running prep", flush=True)
            prep(args, dataset)
        except ImportError:
            print(f"no cache for {dataset} in {out}, running prep with {args.prep_python}", flush=True)
            subprocess.run([os.path.expanduser(args.prep_python), os.path.abspath(__file__), "prep",
                            "--datasets", dataset, "--hann", str(args.hann), "--cache-dir", args.cache_dir,
                            "--data-root", args.data_root, "--jobs", str(args.jobs)], check=True)
    src = os.path.join(out, "source.json")
    if os.path.exists(src) and json.load(open(src))["root"] != dataset_root(args, dataset):
        print(f"warning: {out} was built from {json.load(open(src))['root']}, "
              f"not {dataset_root(args, dataset)}; using the cache", flush=True)
    with open(f) as fh:
        return list(csv.DictReader(fh))


class Recordings:
    """Cached recordings of one dataset with one channel count; epoch() lists non-overlapping valid windows
    (rec, start), load() returns them with n_use channels each."""

    def __init__(self, folder, rows, length, clean, awake_only, max_channels):
        self.x, self.end, self.bad_cs, self.names = [], [], [], []
        self.length, self.clean = length, clean
        for r in rows:
            x = np.load(os.path.join(folder, f"{r['name']}.npy"), mmap_mode="r")
            end = x.shape[0] - length  # last valid start
            n2 = to_float(r.get("n2_onset"))
            if awake_only and np.isfinite(n2):
                end = min(end, int(n2 * SFREQ) - length)
            if end < 0:
                continue
            bad = np.load(os.path.join(folder, f"{r['name']}_bad.npy")).reshape(-1, x.shape[1])
            self.x.append(x)
            self.end.append(end)
            self.bad_cs.append(np.concatenate([np.zeros((1, x.shape[1]), int), np.cumsum(bad, 0)]))
            self.names.append(r["name"])
        self.n_channels = self.x[0].shape[1] if self.x else 0
        self.n_use = min(self.n_channels, max_channels)

    def clean_channels(self, i, s):
        """(len(s), ch) bool: channel without a bad 1 s epoch anywhere in the window starting at s."""
        cs = self.bad_cs[i]
        e0, e1 = s // SFREQ, np.minimum((s + self.length - 1) // SFREQ, len(cs) - 2)
        return cs[e1 + 1] - cs[e0] == 0

    def valid(self, i, s):
        if not self.clean:
            return s
        return s[self.clean_channels(i, s).sum(1) >= self.n_use]

    def epoch(self, rng):
        """(n, 2) array of (recording, start): non-overlapping windows with a random offset per recording."""
        out = []
        for i, end in enumerate(self.end):
            s = self.valid(i, np.arange(rng.integers(0, min(self.length, end + 1)), end + 1, self.length))
            out.append(np.stack([np.full(len(s), i), s], 1))
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=int)

    def load(self, idx, rng):
        """(n, length, n_use) windows; with more channels than n_use, a random subset per window (clean channels
        first under --clean), in montage order."""
        w = np.stack([self.x[i][s:s + self.length] for i, s in idx]).astype(np.float32)
        if self.n_use == self.n_channels:
            return w
        score = rng.random((len(idx), self.n_channels))
        if self.clean:
            for k, (i, s) in enumerate(idx):
                score[k] += ~self.clean_channels(i, np.array([s]))[0]  # bad channels sort last
        pick = np.sort(np.argsort(score, 1)[:, :self.n_use], 1)
        return np.take_along_axis(w, pick[:, None, :], 2)


def standardize(w, context):
    """Windows (n, L+H, ch) -> z-scored with the context statistics, as in panda_muse_clean_eval.py."""
    ctx = w[:, :context]
    return (w - ctx.mean(1, keepdims=True)) / (ctx.std(1, keepdims=True) + 1e-6)


def make_epoch(pools, weights, batch_size, fraction, rng):
    """List of (pool, window index array) batches, shuffled across pools. weights: {dataset: share}."""
    batches = []
    for key, pool in pools.items():
        idx = pool.epoch(rng)
        reps = weights[key[0]] * fraction  # whole copies plus an exact random share of the rest
        rest = idx[rng.permutation(len(idx))[:round((reps - int(reps)) * len(idx))]]
        idx = np.concatenate([idx] * int(reps) + [rest])
        idx = idx[rng.permutation(len(idx))]
        batches += [(key, idx[i:i + batch_size]) for i in range(0, len(idx) - batch_size + 1, batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def set_trainable(model, freeze, unfreeze_last):
    n_layers = len(model.model.encoder.layers)
    keep = {f"model.encoder.layers.{i}." for i in range(n_layers - unfreeze_last, n_layers)}
    for name, p in model.named_parameters():
        if freeze == "encoder":
            p.requires_grad = name.startswith("head.") or any(name.startswith(k) for k in keep)
        elif freeze == "embedder":
            p.requires_grad = not name.startswith("model.encoder.embedder.")
        else:
            p.requires_grad = True


def save(model, path, info):
    for p in model.parameters():  # save_pretrained needs contiguous tensors
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
    model.save_pretrained(path)
    with open(os.path.join(path, "training_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)


def pool_name(key):
    return f"{key[0]}/{key[1]}ch"


def train(args):
    import torch
    if args.panda_src:  # repo folder that contains the `panda` package (not needed if panda is installed)
        sys.path.insert(0, os.path.expanduser(args.panda_src))
    from panda.patchtst.patchtst import PatchTSTConfig, PatchTSTForPrediction

    torch.manual_seed(args.seed)  # also fixes the embedder's unsaved polynomial patch indices (see panda_embed.py)
    torch.set_num_threads(args.threads or os.cpu_count())
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.scratch:
        model = PatchTSTForPrediction(PatchTSTConfig.from_pretrained(args.model))
    else:
        model = PatchTSTForPrediction.from_pretrained(args.model)
    model.to(device)
    L, H = model.config.context_length, model.config.prediction_length
    set_trainable(model, args.freeze, args.unfreeze_last)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{args.model}{' (random init)' if args.scratch else ''}: context {L}, horizon {H}, "
          f"{sum(p.numel() for p in params):,} / {sum(p.numel() for p in model.parameters()):,} trainable, "
          f"device {device}", flush=True)

    weights = args.weights or [1.0] * len(args.datasets)
    assert len(weights) == len(args.datasets), "--weights needs one value per dataset"
    weights = dict(zip(args.datasets, weights))
    clean = set() if args.clean is None else set(args.clean or args.datasets)
    assert clean <= set(args.datasets), "--clean names a dataset that isn't in --datasets"

    # data: per dataset, --val-frac of the train subjects are held out for validation; test splits are never used
    val_sets = [ds for ds in args.datasets if ds in (args.val_datasets or args.datasets)]
    pools, val_pools = {}, {}
    for ds in args.datasets:
        rows = [r for r in read_index(args, ds) if r["split"] != "test"]
        folder = cache_dir(args, ds)
        groups = {"train": rows}
        if ds in val_sets and args.val_frac > 0:
            subs = sorted({r["sub"] for r in rows})
            # own generator per dataset: Muse gets the same val subjects as in the earlier Muse-only runs
            val_subs = set(np.random.default_rng(args.seed).choice(
                subs, size=max(1, round(args.val_frac * len(subs))), replace=False))
            groups = {"train": [r for r in rows if r["sub"] not in val_subs],
                      "val": [r for r in rows if r["sub"] in val_subs]}
            print(f"{ds}: {len(subs) - len(val_subs)} train / {len(val_subs)} val subjects", flush=True)
        for part, part_rows in groups.items():
            for n_ch in sorted({int(r["n_channels"]) for r in part_rows}):
                p = Recordings(folder, [r for r in part_rows if int(r["n_channels"]) == n_ch], L + H, ds in clean,
                               args.awake_only, args.max_channels)
                if not p.x:
                    continue
                (pools if part == "train" else val_pools)[(ds, n_ch)] = p
                hours = sum(len(x) for x in p.x) / SFREQ / 3600
                print(f"{pool_name((ds, n_ch))} {part}: {len(p.x)} recordings, {hours:.1f} h, ~{len(p.epoch(rng)):,} "
                      f"windows per epoch{' (clean)' if ds in clean else ''}, {p.n_use} of {n_ch} channels per window",
                      flush=True)
    if not pools:
        raise SystemExit("no training data")

    # fixed val windows: --val-windows per dataset, shared over its channel-count pools by their window counts
    vr = np.random.default_rng(args.seed + 1)
    val_w = {}  # (dataset, n_ch) -> standardized windows (n, L+H, ch)
    for ds in val_sets:
        keys = [k for k in val_pools if k[0] == ds]
        idx = {k: val_pools[k].epoch(vr) for k in keys}
        n_all = sum(len(i) for i in idx.values())
        for k in keys:
            n = round(args.val_windows * len(idx[k]) / max(1, n_all))
            if n:
                val_w[k] = standardize(val_pools[k].load(idx[k][vr.permutation(len(idx[k]))[:n]], vr), L)
    val_names = sorted({k[0] for k in val_w}, key=args.datasets.index)
    for ds in val_names:
        print(f"{ds} val: {sum(len(w) for k, w in val_w.items() if k[0] == ds)} windows", flush=True)

    metric_names = ["val_mse", "val_mae_n120"] + [f"val_{m}_{ds}" for ds in val_names for m in ("mse", "mae_n120")]
    default_select = f"val_mse_{MUSE}" if MUSE in val_names else "val_mse"
    select = args.select_metric or default_select
    if val_w and select not in metric_names:
        raise SystemExit(f"--select-metric must be one of {metric_names}")

    def evaluate():
        """Val MSE and n-step MAE over all val windows (val_mse, val_mae_n120) and per dataset (val_mse_<ds>, ...)."""
        if not val_w:
            return {}
        model.eval()
        sq, ab = {}, {}  # per dataset: per-window mean squared / absolute (first N_MAE steps) errors
        with torch.no_grad():
            for (ds, _), w in val_w.items():
                for i in range(0, len(w), args.batch_size):
                    c = torch.tensor(w[i:i + args.batch_size, :L], device=device)
                    err = model(past_values=c).prediction_outputs.cpu().numpy() - w[i:i + args.batch_size, L:]
                    sq.setdefault(ds, []).append((err ** 2).mean((1, 2)))
                    ab.setdefault(ds, []).append(np.abs(err[:, :N_MAE]).mean((1, 2)))
        model.train()
        sq, ab = {d: np.concatenate(v) for d, v in sq.items()}, {d: np.concatenate(v) for d, v in ab.items()}
        m = dict(val_mse=float(np.concatenate(list(sq.values())).mean()),
                 val_mae_n120=float(np.concatenate(list(ab.values())).mean()))
        for ds in val_names:
            m[f"val_mse_{ds}"], m[f"val_mae_n120_{ds}"] = float(sq[ds].mean()), float(ab[ds].mean())
        return m

    def wb_val(m):
        """val_mse -> val/mse (combined), val_mse_<ds> -> val/<ds>/mse."""
        out = {}
        for k, v in m.items():
            for metric in ("mse", "mae_n120"):
                if k == f"val_{metric}":
                    out[f"val/{metric}"] = v
                elif k.startswith(f"val_{metric}_"):
                    out[f"val/{k[len(f'val_{metric}_'):]}/{metric}"] = v
        return out

    run = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(os.path.expanduser(args.output_dir), run)
    os.makedirs(out, exist_ok=True)

    # window counts vary slightly with the random offsets, so the schedule length is re-estimated every epoch
    steps_per_epoch = len(make_epoch(pools, weights, args.batch_size, args.epoch_fraction,
                                     np.random.default_rng(args.seed + 2)))
    if steps_per_epoch == 0:
        raise SystemExit("no training batches: fewer windows than --batch-size in every pool")
    total = [steps_per_epoch * args.epochs]
    print(f"{args.epochs} epochs x ~{steps_per_epoch} steps (batch {args.batch_size}) = ~{total[0]} steps",
          flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warm = max(1, int(args.warmup * total[0]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total[0] - warm)))))  # linear warmup, cosine to 0

    # predict-zero (= context mean) baseline per dataset and combined
    zero = {ds: np.concatenate([(w[:, L:] ** 2).mean((1, 2)) for k, w in val_w.items() if k[0] == ds])
            for ds in val_names}
    zero_m = {}
    if zero:
        zero_m["val/zero_mse"] = float(np.concatenate(list(zero.values())).mean())
        zero_m.update({f"val/{ds}/zero_mse": float(v.mean()) for ds, v in zero.items()})
    info = dict(args=vars(args), base_model=args.model, context_length=L, prediction_length=H,
                trainable_params=sum(p.numel() for p in params), steps_per_epoch=steps_per_epoch,
                train_recordings={pool_name(k): len(p.x) for k, p in pools.items()},
                train_hours={pool_name(k): round(sum(len(x) for x in p.x) / SFREQ / 3600, 2) for k, p in pools.items()},
                val_recordings={pool_name(k): len(p.x) for k, p in val_pools.items()},
                val_windows={pool_name(k): len(w) for k, w in val_w.items()}, val_zero_mse=zero_m,
                select_metric=select)
    with open(os.path.join(out, "args.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                        name=run, tags=args.wandb_tags, mode=args.wandb_mode, dir=out,
                        config={**info["args"], **{k: v for k, v in info.items() if k != "args"}})
        wb.define_metric("epoch")
        for k in ("val/*", "train/epoch_mse*", "best/*"):
            wb.define_metric(k, step_metric="epoch")

    fields = ["epoch", "step", "lr", "grad_norm", "train_mse"] + [f"train_mse_{ds}" for ds in args.datasets] + \
        metric_names + ["best_score", "best_epoch", "sec"]
    log_f = open(os.path.join(out, "log.csv"), "w", newline="")
    log = csv.DictWriter(log_f, fieldnames=fields, extrasaction="ignore")
    log.writeheader()

    def report(row, wb_row):
        log.writerow(row)
        log_f.flush()
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        if wb is not None:
            wb.log(wb_row, step=row["step"])

    # epoch 0 = the starting model; the best checkpoint has to beat it (see --save-base)
    m = evaluate()
    best, best_epoch, bad_epochs = m.get(select, math.inf), 0, 0
    print("val: predict-zero (context mean) mse " + " ".join(f"{k[4:]} {v:.4f}" for k, v in zero_m.items())
          + f"; checkpoint-best by {select}", flush=True)
    report(dict(epoch=0, step=0, **m, best_score=best, best_epoch=0),
           {"epoch": 0, **wb_val(m), **zero_m, "best/score": best, "best/epoch": 0})
    if args.save_base:
        save(model, os.path.join(out, "checkpoint-best"), dict(info, epoch=0, step=0, **m))

    model.train()
    t0, step, stopped = time.time(), 0, None
    for epoch in range(1, args.epochs + 1):
        batches = make_epoch(pools, weights, args.batch_size, args.epoch_fraction, rng)
        total[0] = step + len(batches) * (args.epochs - epoch + 1)
        run_loss, run_gn, run_ds, epoch_loss, epoch_ds, t_log = [], [], {}, [], {}, time.time()
        for b, (key, idx) in enumerate(batches, 1):
            w = standardize(pools[key].load(idx, rng), L)
            if args.sign_flip > 0:  # EEG polarity is arbitrary up to the reference
                w *= np.where(rng.random((len(w), 1, w.shape[2])) < args.sign_flip, -1.0, 1.0).astype(np.float32)
            w = torch.tensor(w, device=device)
            loss = model(past_values=w[:, :L], future_values=w[:, L:]).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
            lr = sched.get_last_lr()[0]  # lr used for this step
            opt.step()
            sched.step()
            step += 1
            run_loss.append(loss.item())
            run_gn.append(grad_norm)
            run_ds.setdefault(key[0], []).append(loss.item())
            epoch_loss.append(loss.item())
            epoch_ds.setdefault(key[0], []).append(loss.item())
            if args.log_every and step % args.log_every == 0 and b < len(batches):
                row = dict(epoch=round(epoch - 1 + b / len(batches), 3), step=step, lr=lr,
                           grad_norm=float(np.mean(run_gn)), train_mse=float(np.mean(run_loss)),
                           **{f"train_mse_{d}": float(np.mean(v)) for d, v in run_ds.items()},
                           sec=round(time.time() - t0, 1))
                report(row, {"train/mse": row["train_mse"], "train/lr": lr, "train/grad_norm": row["grad_norm"],
                             "train/epoch": row["epoch"],
                             "train/windows_per_sec": len(run_loss) * args.batch_size / (time.time() - t_log),
                             **{f"train/mse_{d}": float(np.mean(v)) for d, v in run_ds.items()}})
                run_loss, run_gn, run_ds, t_log = [], [], {}, time.time()

        m = evaluate()
        score = m.get(select, math.inf)
        improved = not val_w or score < best - args.min_delta
        if improved:
            best, best_epoch, bad_epochs = score, epoch, 0
            save(model, os.path.join(out, "checkpoint-best"),
                 dict(info, epoch=epoch, step=step, train_mse=float(np.mean(epoch_loss)), **m))
        else:
            bad_epochs += 1
        if args.save_every and epoch % args.save_every == 0:
            save(model, os.path.join(out, f"checkpoint-epoch{epoch}"), dict(info, epoch=epoch, step=step, **m))
        train_ds = {d: float(np.mean(v)) for d, v in epoch_ds.items()}
        row = dict(epoch=epoch, step=step, lr=sched.get_last_lr()[0], train_mse=float(np.mean(epoch_loss)),
                   **{f"train_mse_{d}": v for d, v in train_ds.items()}, **m,
                   best_score=best, best_epoch=best_epoch, sec=round(time.time() - t0, 1))
        report(row, {"epoch": epoch, "train/epoch_mse": row["train_mse"],
                     **{f"train/epoch_mse_{d}": v for d, v in train_ds.items()}, **wb_val(m),
                     "best/score": best, "best/epoch": best_epoch, "val/epochs_since_best": bad_epochs})
        if improved:
            print(f"  new best {select} {best:.4f} -> {out}/checkpoint-best", flush=True)
        if args.patience and bad_epochs >= args.patience:
            stopped = epoch
            print(f"early stopping: no {select} improvement > {args.min_delta} for {bad_epochs} epochs "
                  f"(best {best:.4f} at epoch {best_epoch})", flush=True)
            break

    last = stopped or args.epochs
    save(model, os.path.join(out, "checkpoint-final"), dict(info, epoch=last, step=step))
    log_f.close()
    summary = dict(select_metric=select, best_score=best, best_epoch=best_epoch, last_epoch=last, stopped_early=stopped is not None,
                   steps=step, val_zero_mse=zero_m, minutes=round((time.time() - t0) / 60, 1),
                   best_checkpoint=os.path.join(out, "checkpoint-best") if os.path.exists(
                       os.path.join(out, "checkpoint-best")) else None)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if wb is not None:
        wb.summary.update(summary)
        if args.wandb_log_model and summary["best_checkpoint"]:
            art = wandb.Artifact(f"panda-{run}", type="model", metadata=summary)
            art.add_dir(summary["best_checkpoint"])
            wb.log_artifact(art, aliases=["best"])
        wb.finish()
    print(f"done: {out}  best {select} {best:.4f} at epoch {best_epoch}"
          f"{' (the starting model; pass --save-base to keep it as checkpoint-best)' if best_epoch == 0 else ''}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add_paths(p):
        p.add_argument("--datasets", nargs="+", default=DATASETS, help="BIDS dataset folders under --data-root")
        p.add_argument("--data-root", default=WORK, help="folder holding the dataset folders")
        p.add_argument("--cache-dir", default=os.path.join(WORK, "cache_dir", "panda_train"),
                       help="where preprocessed recordings are cached")
        p.add_argument("--hann", type=int, default=15, help="Hann smoothing taps (0 = none)")
        p.add_argument("--jobs", type=int, default=1, help="recordings prepared in parallel")

    p = sp.add_parser("scan", help="list each dataset's EEG files and channels (no signal loading)")
    add_paths(p)
    p = sp.add_parser("prep", help="cache preprocessed recordings (needs mne)")
    add_paths(p)

    t = sp.add_parser("train", help="fine-tune Panda (panda env)")
    add_paths(t)
    t.add_argument("--prep-python", default="~/miniconda3/envs/dynamix/bin/python",
                   help="interpreter with mne, used for prep if mne is not importable here")
    t.add_argument("--panda-src", default=REPO,
                   help="folder containing the panda package ('' if panda is installed)")
    t.add_argument("--model", default=MODEL, help="HF id or local checkpoint to start from")
    t.add_argument("--scratch", action="store_true", help="random init with --model's config (train from scratch)")
    t.add_argument("--weights", type=float, nargs="+", help="per dataset: share of its windows per epoch (default 1)")
    t.add_argument("--clean", nargs="*", metavar="DS",
                   help="only windows without bad 1 s epochs: all datasets, or only the ones listed")
    t.add_argument("--max-channels", type=int, default=8,
                   help="EEG channels per window; recordings with more get a random subset each draw")
    t.add_argument("--awake-only", action="store_true", help="only windows ending before N2 onset (where known)")
    t.add_argument("--val-datasets", nargs="+", metavar="DS",
                   help="datasets that get held-out val subjects (default: all in --datasets)")
    t.add_argument("--val-frac", type=float, default=0.1, help="fraction of each val dataset's train subjects held out")
    t.add_argument("--val-windows", type=int, default=512, help="fixed val windows per dataset")
    t.add_argument("--select-metric",
                   help="lower is better; picks checkpoint-best and drives early stopping: val_mse / val_mae_n120 "
                        "(all val data) or val_mse_<ds> / val_mae_n120_<ds> (default: val_mse_nm000287, i.e. Muse)")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--epoch-fraction", type=float, default=1.0,
                   help="random share of the windows used per epoch (e.g. 0.1 for short epochs on CPU)")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--warmup", type=float, default=0.05, help="fraction of all steps")
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--freeze", default="none", choices=["none", "embedder", "encoder"],
                   help="encoder: train only the head (+ --unfreeze-last layers)")
    t.add_argument("--unfreeze-last", type=int, default=0, help="with --freeze encoder: last N layers stay trainable")
    t.add_argument("--sign-flip", type=float, default=0.0, help="probability of flipping a channel's sign")
    t.add_argument("--log-every", type=int, default=50, help="steps between train-loss lines (0 = per epoch only)")
    t.add_argument("--save-every", type=int, default=0, help="also keep checkpoint-epoch<k> every k epochs (0 = off)")
    t.add_argument("--patience", type=int, default=3,
                   help="early stopping: epochs without --select-metric improvement before stopping (0 = off)")
    t.add_argument("--min-delta", type=float, default=0.0, help="minimum val_mse decrease that counts as improvement")
    t.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    t.add_argument("--wandb-project", default="panda-eeg")
    t.add_argument("--wandb-entity")
    t.add_argument("--wandb-group")
    t.add_argument("--wandb-tags", nargs="+")
    t.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                   help="offline: log locally, upload later with `wandb sync`")
    t.add_argument("--wandb-log-model", action="store_true", help="upload checkpoint-best as a wandb artifact")
    t.add_argument("--save-base", action="store_true", help="save the base model as checkpoint-best first")
    t.add_argument("--output-dir", default=os.path.join(WORK, "save_dir", "panda_muse"))
    t.add_argument("--run-name")
    t.add_argument("--device", help="cpu / cuda / cuda:1 / mps (default: cuda if available)")
    t.add_argument("--threads", type=int)
    t.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if args.cmd in ("scan", "prep"):
        for ds in args.datasets:
            (scan if args.cmd == "scan" else prep)(args, ds)
    else:
        train(args)


if __name__ == "__main__":
    main()
