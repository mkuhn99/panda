"""Fine-tune (or train from scratch) Panda on Muse EEG (nm000287) and, optionally, other EEG datasets.

Two stages, since Panda (transformers==4.40.2, numpy<2) and mne often live in different envs:
  prep   (any env with mne + scipy): each recording -> 0.5-40 Hz, uV, 128 Hz, Hann 15 (as in
         panda_muse_clean_eval.py), saved as <cache-dir>/<dataset>_hann<k>/<rec>.npy (T, ch) plus a 1 s bad-epoch
         mask (artifact rule of dynamix_muse_clean_eval.py) and an index.csv.
  train  (panda env): epochs over non-overlapping (context 512 + horizon 128) windows of every recording, with a
         new random offset each epoch; each window is standardized with its context statistics (as in the eval),
         MSE on the next 128 samples. If the cache is missing, `prep` runs in-process when mne is importable,
         otherwise through --prep-python.

Datasets and their roots (all paths are arguments, so the script runs on any machine):
  muse      --muse-root      BIDS folder of nm000287 (sub-*/sub-*_sessions.tsv with a `split` column);
                             only the train split is used, the test split stays untouched for the eval
  eegmmidb  --eegmmidb-root  folder of PhysioNet eegmmidb *.edf files (Fz Pz Oz T7)
  sleepedf  --sleepedf-root  folder of Sleep-EDF *-PSG.edf files (Fpz-Cz Pz-Oz)
Each batch comes from one dataset (channel counts differ). --weights scales how many of a dataset's windows go
into an epoch (0.5 = a random half, 2 = each window twice).
Note: the eegmmidb files here are the subjects dynamix_eegmmidb_eval.py tests on, so training on them makes that
eval in-sample.
Validation: a held-out fraction of Muse *train* subjects; MSE and MAE over the first 120 steps (the eval's n-step
MAE) on a fixed set of windows, before training (zero-shot) and after every epoch. The weights of the epoch with
the lowest val MSE go to checkpoint-best; training stops early after --patience epochs without improvement.
Outputs per run: checkpoint-best, checkpoint-final, log.csv (train / val loss, lr, grad norm per step and epoch),
summary.json (best epoch, early stop). With --wandb the same statistics go to Weights & Biases.

Default paths are relative to the folder that contains the panda repo; override them on other machines.
Checkpoints are saved with save_pretrained, so they plug straight into the existing eval scripts:
    python scripts/panda/train_muse.py prep --datasets muse --muse-root /data/nm000287 --cache-dir /scratch/panda_cache
    python scripts/panda/train_muse.py train --epochs 3 --clean --muse-root /data/nm000287 --cache-dir /scratch/panda_cache
    python scripts/panda/train_muse.py train --epochs 2 --datasets muse sleepedf --weights 1 0.3 --sleepedf-root ~/sleep-edfx
    python scripts/panda/train_muse.py train --epochs 5 --epoch-fraction 0.2 --freeze encoder --unfreeze-last 2 --lr 3e-4
    python scripts/panda/train_muse.py train --epochs 30 --patience 4 --wandb --wandb-project panda-eeg --wandb-log-model
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

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # panda repo
WORK = os.path.dirname(REPO)  # folder holding the panda repo and the data (defaults only; all are arguments)
MODEL = "GilpinLab/panda"
SFREQ = 128
N_MAE = 120
DATASETS = ("muse", "eegmmidb", "sleepedf")
CHANNELS = {"eegmmidb": ["Fz", "Pz", "Oz", "T7"], "sleepedf": ["EEG Fpz-Cz", "EEG Pz-Oz"]}
P2P_UV, FLAT_UV, Z_MAX = 100.0, 0.5, 3.0  # artifact rule of dynamix_muse_clean_eval.py


def cache_dir(args, dataset):
    return os.path.join(os.path.expanduser(args.cache_dir), f"{dataset}_hann{args.hann}")


def dataset_root(args, dataset):
    return os.path.expanduser(getattr(args, f"{dataset}_root"))


def to_float(v):
    try:
        return float(v)
    except ValueError:
        return math.nan


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
    """(n_epochs,) bool for 1 s epochs of the 0.5-40 Hz signal x (T, ch) in uV: peak-to-peak, flat, or robust
    outlier in log variance / log 20-40 Hz power."""
    from scipy.signal import welch
    n = len(x) // SFREQ
    ep = x[:n * SFREQ].reshape(n, SFREQ, -1)
    sd = ep.std(1)
    f, p = welch(ep, fs=SFREQ, nperseg=SFREQ, axis=1)
    emg = p[:, (f >= 20) & (f <= 40)].sum(1)
    bad = (np.ptp(ep, axis=1) > P2P_UV) | (sd < FLAT_UV)
    bad |= robust_z(np.log(sd + 1e-12)) > Z_MAX
    bad |= robust_z(np.log(emg + 1e-12)) > Z_MAX
    return bad.any(1)


def read_raw(f, channels=None):
    """0.5-40 Hz, 128 Hz, (T, ch) in uV."""
    import mne
    raw = (mne.io.read_raw_brainvision if f.endswith(".vhdr") else mne.io.read_raw_edf)(f, preload=True)
    raw.rename_channels(lambda c: c.strip("."))  # eegmmidb labels are padded, e.g. "Fz.."
    if channels:
        raw.pick(channels)
    raw.filter(0.5, 40.0)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ)
    return raw.get_data().T * 1e6


def muse_recordings(root):
    for f in sorted(glob.glob(os.path.join(root, "sub-*", "sub-*_sessions.tsv"))):
        sub = os.path.basename(f).split("_")[0][4:]
        with open(f) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                ses = r["session_id"][4:]
                vhdr = glob.glob(os.path.join(root, f"sub-{sub}", f"ses-{ses}", "eeg", "*_eeg.vhdr"))
                if not vhdr:
                    print(f"muse: no .vhdr for sub-{sub} ses-{ses}, skipped", flush=True)
                    continue
                yield (f"sub-{sub}_ses-{ses}", lambda f=vhdr[0]: read_raw(f),
                       dict(sub=sub, ses=ses, split=r.get("split", "train"), n2_onset=r.get("n2_onset", "")))


def edf_recordings(files, channels, subject):
    for f in sorted(files):
        name = os.path.basename(f).split(".")[0]
        yield name, lambda f=f: read_raw(f, channels), dict(sub=subject(name), ses="", split="train", n2_onset="")


def prep(args, dataset):
    import mne
    mne.set_log_level("ERROR")
    root = dataset_root(args, dataset)
    recs = {
        "muse": lambda: muse_recordings(root),
        "eegmmidb": lambda: edf_recordings(glob.glob(os.path.join(root, "*.edf")), CHANNELS["eegmmidb"],
                                           lambda n: n[1:4]),
        "sleepedf": lambda: edf_recordings(glob.glob(os.path.join(root, "**", "*-PSG.edf"), recursive=True),
                                           CHANNELS["sleepedf"], lambda n: n[:5]),
    }[dataset]()
    out = cache_dir(args, dataset)
    os.makedirs(out, exist_ok=True)
    rows = []
    for name, loader, meta in recs:
        f, fb = os.path.join(out, f"{name}.npy"), os.path.join(out, f"{name}_bad.npy")
        if not (os.path.exists(f) and os.path.exists(fb)):
            x = loader()
            np.save(fb, bad_epochs(x))  # on the unsmoothed signal, as in the eval
            np.save(f, smooth(x, args.hann).astype(np.float32))
        x = np.load(f, mmap_mode="r")
        rows.append(dict(name=name, **meta, n_samples=x.shape[0], n_channels=x.shape[1]))
        print(f"{dataset}: {name} {x.shape}", flush=True)
    if not rows:
        raise SystemExit(f"no {dataset} recordings found under {root}")
    with open(os.path.join(out, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "source.json"), "w") as fh:
        json.dump(dict(root=root, hann=args.hann), fh)
    print(f"{dataset}: {len(rows)} recordings from {root} -> {out}", flush=True)


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
                            f"--{dataset}-root", dataset_root(args, dataset)], check=True)
    src = os.path.join(out, "source.json")
    if os.path.exists(src) and json.load(open(src))["root"] != dataset_root(args, dataset):
        print(f"warning: {out} was built from {json.load(open(src))['root']}, "
              f"not {dataset_root(args, dataset)}; using the cache", flush=True)
    with open(f) as fh:
        return list(csv.DictReader(fh))


class Recordings:
    """Cached recordings of one dataset; epoch() lists non-overlapping valid windows (rec, start)."""

    def __init__(self, folder, rows, length, clean, awake_only):
        self.x, self.end, self.bad_cs = [], [], []
        self.length, self.clean = length, clean
        for r in rows:
            x = np.load(os.path.join(folder, f"{r['name']}.npy"), mmap_mode="r")
            end = x.shape[0] - length  # last valid start
            n2 = to_float(r["n2_onset"])
            if awake_only and np.isfinite(n2):
                end = min(end, int(n2 * SFREQ) - length)
            if end < 0:
                continue
            bad = np.load(os.path.join(folder, f"{r['name']}_bad.npy"))
            self.x.append(x)
            self.end.append(end)
            self.bad_cs.append(np.concatenate([[0], np.cumsum(bad)]))
        self.n_channels = self.x[0].shape[1] if self.x else 0

    def valid(self, i, s):
        if not self.clean:
            return s
        cs = self.bad_cs[i]  # no bad 1 s epoch anywhere in the window
        e0, e1 = s // SFREQ, np.minimum((s + self.length - 1) // SFREQ, len(cs) - 2)
        return s[cs[e1 + 1] - cs[e0] == 0]

    def epoch(self, rng):
        """(n, 2) array of (recording, start): non-overlapping windows with a random offset per recording."""
        out = []
        for i, end in enumerate(self.end):
            s = self.valid(i, np.arange(rng.integers(0, min(self.length, end + 1)), end + 1, self.length))
            out.append(np.stack([np.full(len(s), i), s], 1))
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=int)

    def load(self, idx):
        return np.stack([self.x[i][s:s + self.length] for i, s in idx]).astype(np.float32)


def standardize(w, context):
    """Windows (n, L+H, ch) -> z-scored with the context statistics, as in panda_muse_clean_eval.py."""
    ctx = w[:, :context]
    return (w - ctx.mean(1, keepdims=True)) / (ctx.std(1, keepdims=True) + 1e-6)


def make_epoch(pools, weights, batch_size, fraction, rng):
    """List of (dataset, window index array) batches, shuffled across datasets."""
    batches = []
    for (ds, pool), w in zip(pools.items(), weights):
        idx = pool.epoch(rng)
        reps = w * fraction  # whole copies plus an exact random share of the rest
        rest = idx[rng.permutation(len(idx))[:round((reps - int(reps)) * len(idx))]]
        idx = np.concatenate([idx] * int(reps) + [rest])
        idx = idx[rng.permutation(len(idx))]
        batches += [(ds, idx[i:i + batch_size]) for i in range(0, len(idx) - batch_size + 1, batch_size)]
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


def train(args):
    import torch
    if args.panda_src:  # repo folder that contains the `panda` package (not needed if panda is installed)
        sys.path.insert(0, os.path.expanduser(args.panda_src))
    from panda.patchtst.patchtst import PatchTSTConfig, PatchTSTForPrediction

    torch.manual_seed(args.seed)
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

    # data: muse train subjects split into train / val by subject; other datasets are train only
    weights = args.weights or [1.0] * len(args.datasets)
    assert len(weights) == len(args.datasets), "--weights needs one value per dataset"
    pools, val = {}, None
    for ds in args.datasets:
        rows = [r for r in read_index(args, ds) if r["split"] == "train"]
        folder = cache_dir(args, ds)
        if ds == "muse" and args.val_frac > 0:
            subs = sorted({r["sub"] for r in rows})
            val_subs = set(rng.choice(subs, size=max(1, round(args.val_frac * len(subs))), replace=False))
            val = Recordings(folder, [r for r in rows if r["sub"] in val_subs], L + H, args.clean, args.awake_only)
            rows = [r for r in rows if r["sub"] not in val_subs]
            print(f"muse: {len(subs) - len(val_subs)} train / {len(val_subs)} val subjects", flush=True)
        pools[ds] = Recordings(folder, rows, L + H, args.clean, args.awake_only)
        print(f"{ds}: {len(pools[ds].x)} recordings, ~{len(pools[ds].epoch(rng)):,} windows per epoch, "
              f"{pools[ds].n_channels} channels", flush=True)

    val_w = None
    if val is not None and val.x:
        vr = np.random.default_rng(args.seed + 1)
        idx = val.epoch(vr)
        val_w = standardize(val.load(idx[vr.permutation(len(idx))[:args.val_windows]]), L)

    def evaluate():
        if val_w is None:
            return {}
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(val_w), args.batch_size):
                c = torch.tensor(val_w[i:i + args.batch_size, :L], device=device)
                preds.append(model(past_values=c).prediction_outputs.cpu().numpy())
        model.train()
        err = np.concatenate(preds) - val_w[:, L:]
        return dict(val_mse=float((err ** 2).mean()), val_mae_n120=float(np.abs(err[:, :N_MAE]).mean()))

    run = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(os.path.expanduser(args.output_dir), run)
    os.makedirs(out, exist_ok=True)

    # window counts vary slightly with the random offsets, so the schedule length is re-estimated every epoch
    steps_per_epoch = len(make_epoch(pools, weights, args.batch_size, args.epoch_fraction,
                                     np.random.default_rng(args.seed + 2)))
    if steps_per_epoch == 0:
        raise SystemExit("no training batches: fewer windows than --batch-size")
    total = [steps_per_epoch * args.epochs]
    print(f"{args.epochs} epochs x ~{steps_per_epoch} steps (batch {args.batch_size}) = ~{total[0]} steps",
          flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warm = max(1, int(args.warmup * total[0]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total[0] - warm)))))  # linear warmup, cosine to 0

    zero_mse = float((val_w[:, L:] ** 2).mean()) if val_w is not None else math.nan
    info = dict(args=vars(args), base_model=args.model, context_length=L, prediction_length=H,
                trainable_params=sum(p.numel() for p in params), steps_per_epoch=steps_per_epoch,
                train_recordings={d: len(p.x) for d, p in pools.items()},
                val_recordings=len(val.x) if val is not None else 0,
                val_windows=0 if val_w is None else len(val_w), val_zero_mse=zero_mse)
    with open(os.path.join(out, "args.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                        name=run, tags=args.wandb_tags, mode=args.wandb_mode, dir=out,
                        config={**info["args"], **{k: v for k, v in info.items() if k != "args"}})
        wb.define_metric("epoch")
        for k in ("val/*", "train/epoch_mse", "best/*"):
            wb.define_metric(k, step_metric="epoch")

    fields = ["epoch", "step", "lr", "grad_norm", "train_mse", "val_mse", "val_mae_n120", "best_val_mse",
              "best_epoch", "sec"]
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
    best, best_epoch, bad_epochs = m.get("val_mse", math.inf), 0, 0
    print(f"val: predict-zero (context mean) mse {zero_mse:.4f}", flush=True)
    report(dict(epoch=0, step=0, **m, best_val_mse=best, best_epoch=0),
           {"epoch": 0, "val/mse": m.get("val_mse"), "val/mae_n120": m.get("val_mae_n120"),
            "val/zero_mse": zero_mse, "best/val_mse": best, "best/epoch": 0})
    if args.save_base:
        save(model, os.path.join(out, "checkpoint-best"), dict(info, epoch=0, step=0, **m))

    model.train()
    t0, step, stopped = time.time(), 0, None
    for epoch in range(1, args.epochs + 1):
        batches = make_epoch(pools, weights, args.batch_size, args.epoch_fraction, rng)
        total[0] = step + len(batches) * (args.epochs - epoch + 1)
        run_loss, run_gn, run_ds, epoch_loss, t_log = [], [], {}, [], time.time()
        for b, (ds, idx) in enumerate(batches, 1):
            w = standardize(pools[ds].load(idx), L)
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
            run_ds.setdefault(ds, []).append(loss.item())
            epoch_loss.append(loss.item())
            if args.log_every and step % args.log_every == 0 and b < len(batches):
                row = dict(epoch=round(epoch - 1 + b / len(batches), 3), step=step, lr=lr,
                           grad_norm=float(np.mean(run_gn)), train_mse=float(np.mean(run_loss)),
                           sec=round(time.time() - t0, 1))
                report(row, {"train/mse": row["train_mse"], "train/lr": lr, "train/grad_norm": row["grad_norm"],
                             "train/epoch": row["epoch"],
                             "train/windows_per_sec": len(run_loss) * args.batch_size / (time.time() - t_log),
                             **{f"train/mse_{d}": float(np.mean(v)) for d, v in run_ds.items()}})
                run_loss, run_gn, run_ds, t_log = [], [], {}, time.time()

        m = evaluate()
        val_mse = m.get("val_mse", math.inf)
        improved = val_w is None or val_mse < best - args.min_delta
        if improved:
            best, best_epoch, bad_epochs = val_mse, epoch, 0
            save(model, os.path.join(out, "checkpoint-best"),
                 dict(info, epoch=epoch, step=step, train_mse=float(np.mean(epoch_loss)), **m))
        else:
            bad_epochs += 1
        if args.save_every and epoch % args.save_every == 0:
            save(model, os.path.join(out, f"checkpoint-epoch{epoch}"), dict(info, epoch=epoch, step=step, **m))
        row = dict(epoch=epoch, step=step, lr=sched.get_last_lr()[0], train_mse=float(np.mean(epoch_loss)), **m,
                   best_val_mse=best, best_epoch=best_epoch, sec=round(time.time() - t0, 1))
        report(row, {"epoch": epoch, "train/epoch_mse": row["train_mse"], "val/mse": m.get("val_mse"),
                     "val/mae_n120": m.get("val_mae_n120"), "best/val_mse": best, "best/epoch": best_epoch,
                     "val/epochs_since_best": bad_epochs})
        if improved:
            print(f"  new best val_mse {best:.4f} -> {out}/checkpoint-best", flush=True)
        if args.patience and bad_epochs >= args.patience:
            stopped = epoch
            print(f"early stopping: no val_mse improvement > {args.min_delta} for {bad_epochs} epochs "
                  f"(best {best:.4f} at epoch {best_epoch})", flush=True)
            break

    last = stopped or args.epochs
    save(model, os.path.join(out, "checkpoint-final"), dict(info, epoch=last, step=step))
    log_f.close()
    summary = dict(best_val_mse=best, best_epoch=best_epoch, last_epoch=last, stopped_early=stopped is not None,
                   steps=step, val_zero_mse=zero_mse, minutes=round((time.time() - t0) / 60, 1),
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
    print(f"done: {out}  best val_mse {best:.4f} at epoch {best_epoch}"
          f"{' (the starting model; pass --save-base to keep it as checkpoint-best)' if best_epoch == 0 else ''}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add_paths(p):
        p.add_argument("--datasets", nargs="+", default=["muse"], choices=DATASETS)
        p.add_argument("--muse-root", default=os.path.join(WORK, "nm000287"), help="nm000287 BIDS folder")
        p.add_argument("--eegmmidb-root", default=os.path.join(WORK, "eegmmidb"), help="folder of eegmmidb .edf")
        p.add_argument("--sleepedf-root", default="~/sleep-edfx/fresh", help="folder of Sleep-EDF *-PSG.edf")
        p.add_argument("--cache-dir", default=os.path.join(WORK, "cache_dir", "panda_train"),
                       help="where preprocessed recordings are cached")
        p.add_argument("--hann", type=int, default=15, help="Hann smoothing taps (0 = none)")

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
    t.add_argument("--clean", action="store_true", help="only windows without bad 1 s epochs")
    t.add_argument("--awake-only", action="store_true", help="Muse: only windows ending before N2 onset")
    t.add_argument("--val-frac", type=float, default=0.1, help="fraction of Muse train subjects held out")
    t.add_argument("--val-windows", type=int, default=512)
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
                   help="early stopping: epochs without val_mse improvement before stopping (0 = off)")
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
    if args.cmd == "prep":
        for ds in args.datasets:
            prep(args, ds)
    else:
        train(args)


if __name__ == "__main__":
    main()
